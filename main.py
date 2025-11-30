from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os

import mimetypes
import torch
import collections


from config import config, Args
from utils.util import pil_to_frame, bytes_to_pil
from utils.connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline

from utils.https_proxy import run_https_proxy

from utils.speech2text import SpeechToText
from fastapi import UploadFile, File

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
# logging.basicConfig(level=logging.DEBUG)

stt = SpeechToText()

class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.init_app()
        self.user_modes = {} 


    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        
        
        @self.app.post("/api/stt")
        async def speech_to_text(file: UploadFile = File(...)):
            audio_bytes = await file.read()
            text = stt.transcribe(audio_bytes)
            return {"text": text}
        
        @self.app.post("/api/ip_ref_image")
        async def upload_ip_ref_image(file: UploadFile = File(...)):
            """Upload IP-Adapter reference image"""
            try:
                image_bytes = await file.read()
                image_pil = bytes_to_pil(image_bytes)
                
                # Update the pipeline's IP-Adapter reference image
                if pipeline.stream.use_ip_adapter and pipeline.stream.ip_adapter is not None:
                    pipeline.update_ip_ref_image(image_pil)
                    # Verify update success
                    if pipeline.stream.ip_adapter.ip_tokens is not None:
                        print(f"[IP-Adapter] ✅ Reference image uploaded, ip_tokens shape: {pipeline.stream.ip_adapter.ip_tokens.shape}")
                        return JSONResponse({
                            "status": "success",
                            "message": "IP-Adapter reference image updated successfully"
                        })
                    else:
                        print("[IP-Adapter] ⚠️ Reference image uploaded but ip_tokens is None")
                        return JSONResponse({
                            "status": "warning",
                            "message": "Image uploaded but ip_tokens is None"
                        })
                else:
                    return JSONResponse({
                        "status": "error",
                        "message": "IP-Adapter is not enabled"
                    }, status_code=400)
            except Exception as e:
                logging.error(f"Error uploading IP ref image: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse({
                    "status": "error",
                    "message": str(e)
                }, status_code=500)
        
        @self.app.post("/api/ip_ref_image/clear")
        async def clear_ip_ref_image():
            """Clear IP-Adapter reference image"""
            try:
                if pipeline.stream.use_ip_adapter and pipeline.stream.ip_adapter is not None:
                    pipeline.stream.set_ip_adapter_image(None)
                    print("[IP-Adapter] ✅ Reference image cleared - IP-Adapter disabled")
                    return JSONResponse({
                        "status": "success",
                        "message": "IP-Adapter reference image cleared"
                    })
                else:
                    return JSONResponse({
                        "status": "error",
                        "message": "IP-Adapter is not enabled"
                    }, status_code=400)
            except Exception as e:
                logging.error(f"Error clearing IP ref image: {e}")
                return JSONResponse({
                    "status": "error",
                    "message": str(e)
                }, status_code=500)
        
        
        

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        import asyncio

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")

            self.user_modes[user_id] = "full"
            print(f"🧠 Initialized mode for {user_id} = full")

            try:
                while True:
                    try:
                        data = await asyncio.wait_for(self.conn_manager.receive_json(user_id), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    # Received mode message
                    if "mode" in data:
                        self.user_modes[user_id] = data["mode"]
                        await self.conn_manager.send_json(user_id, {
                            "status": "mode_changed",
                            "mode": self.user_modes[user_id]
                        })
                        print(f"🎯 User {user_id} switched mode → {self.user_modes[user_id]}")
                        continue

                    # Regular frame logic
                    if data.get("status") == "next_frame":
                        info = pipeline.Info()
                        params = await self.conn_manager.receive_json(user_id)
                        if not params or not isinstance(params, dict):
                            logging.warning(f"⚠️ Received invalid params: {params}")
                            continue
                        current_mode = self.user_modes.get(user_id, "full")
                        params["mode"] = current_mode

                        params = pipeline.InputParams(**params)
                        params = SimpleNamespace(**params.dict())
                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if not image_data:
                                await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                continue
                            params.image = bytes_to_pil(image_data)
                        await self.conn_manager.update_data(user_id, params)
            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id}")
                await self.conn_manager.disconnect(user_id)



        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                async def generate():
                    while True:
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        
                        # Force sync current user mode
                        current_mode = self.user_modes.get(user_id, "full")
                        setattr(params, "mode", current_mode)
                        
                        image = pipeline.predict(params)
                        if image is None:
                            continue
                        frame = pil_to_frame(image)
                        yield frame

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = pipeline.Info.schema()
            info = pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        if not os.path.exists("public"):
            os.makedirs("public")

        self.app.mount(
            "/", StaticFiles(directory="./frontend/public", html=True), name="public"
        )
        
def get_ip():
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print("this node real ip:", ip)
    return ip


def start_https_proxy(ip):
    """
    Run HTTPS proxy in background
    """
    print("🌐 Launching HTTPS proxy server...")
    run_https_proxy(
        target=f"http://{ip}:{config.port}",  
        port=8443
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16
pipeline = Pipeline(config, device, torch_dtype)
app = App(config, pipeline).app

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    
    ip = get_ip()

    proxy_process = multiprocessing.Process(target=start_https_proxy, args=(ip,), daemon=True)
    proxy_process.start()
    
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
