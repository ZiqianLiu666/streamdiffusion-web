import asyncio
import ssl
from aiohttp import web, ClientSession, WSMsgType

async def create_proxy_app(target: str):
    """
    Create aiohttp HTTPS proxy app supporting HTTP, WebSocket, MJPEG forwarding.
    """

    async def proxy_handler(request):
        # ---------- WebSocket handling ----------
        if request.headers.get("Upgrade", "").lower() == "websocket":
            ws_server = web.WebSocketResponse()
            await ws_server.prepare(request)

            async with ClientSession() as session:
                async with session.ws_connect(f"{target}{request.path_qs}") as ws_client:

                    async def ws_to_ws(src, dst):
                        async for msg in src:
                            if msg.type == WSMsgType.TEXT:
                                await dst.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await dst.send_bytes(msg.data)
                            elif msg.type == WSMsgType.CLOSE:
                                await dst.close()

                    await asyncio.gather(
                        ws_to_ws(ws_server, ws_client),
                        ws_to_ws(ws_client, ws_server)
                    )

            return ws_server

        # ---------- HTTP request handling ----------
        async with ClientSession() as session:
            async with session.request(
                request.method,
                f"{target}{request.path_qs}",
                headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
                data=await request.read(),
            ) as resp:
                # MJPEG streaming forward
                if resp.headers.get("Content-Type", "").startswith("multipart/x-mixed-replace"):
                    response = web.StreamResponse(status=resp.status, headers=resp.headers)
                    await response.prepare(request)
                    async for chunk in resp.content.iter_chunked(1024):
                        await response.write(chunk)
                    return response
                # Regular HTTP request
                body = await resp.read()
                return web.Response(status=resp.status, headers=resp.headers, body=body)

    app = web.Application()
    app.router.add_route('*', '/{path:.*}', proxy_handler)
    return app


def run_https_proxy(target: str = "http://localhost:7860", port: int = 8443,
                    cert_path: str = "utils/certs/cert.pem", key_path: str = "utils/certs/key.pem"):
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(cert_path, key_path)

    app = asyncio.run(create_proxy_app(target))
    
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"✅ Proxy running at: https://{ip}:{port}/  (forwarding to {target})")
    
    web.run_app(app, host='0.0.0.0', port=port, ssl_context=ssl_context, handle_signals=False)


# # If running script directly, start by default
# if __name__ == "__main__":
#     run_https_proxy(target="http://137.194.132.113:7860", port=8443)