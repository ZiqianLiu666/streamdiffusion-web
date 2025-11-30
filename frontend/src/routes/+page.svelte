<script lang="ts">
  import { onMount } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import InputRange from '$lib/components/InputRange.svelte';
  import TextArea from '$lib/components/TextArea.svelte';
  import { FieldType } from '$lib/types';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';
  import { pipelineValues } from '$lib/store';
  import { get } from 'svelte/store';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  let ipRefImagePreview: string | null = null;
  let ipRefImageFile: File | null = null;
  
  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch('/api/settings').then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    console.log(pipelineParams);
    toggleQueueChecker(true);
  }


  function updateMode(newMode: string) {
    pipelineValues.update(v => ({ ...v, mode: newMode }));
    console.log('🧩 Mode switched to:', newMode);

    const socket = (window as any).globalWs as WebSocket | undefined;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ mode: newMode }));
      console.log(`🌐 Sent mode change to backend: ${newMode}`);
    } else {
      console.warn('⚠️ Not connected yet — click Start first!');
    }
  }


  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch('/api/queue').then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }

  function getSreamdata() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }
  let disabled = false;
  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();
        }
        disabled = true;
        await lcmLiveActions.start(getSreamdata);
        disabled = false;
        toggleQueueChecker(false);

        // Send current mode to backend after connection is established
        const current = getPipelineValues();
        const modeToSend = current?.mode ?? 'full';
        lcmLiveActions.send({ mode: modeToSend });
        console.log('🚀 Sent initial mode after connect:', modeToSend);

      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        lcmLiveActions.stop();
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      toggleQueueChecker(true);
    }
  }




  // 🎤 === Speech to Text ===
  let recording = false;
  let mediaRecorder: MediaRecorder;
  let audioChunks: BlobPart[] = [];

  async function toggleRecording() {
    if (!recording) {
      // Start recording
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', blob, 'speech.wav');

        try {
          const res = await fetch('/api/stt', { method: 'POST', body: formData });
          const data = await res.json();
          const recognized = data.text?.trim() || '';
          console.log('🎙️ Recognized:', recognized);

          pipelineValues.update((current) => ({
            ...current,
            prompt: recognized
          }));

          console.log('✅ Prompt store updated');

          const textarea = document.querySelector('textarea[placeholder*="prompt"]');
          if (textarea) {
            textarea.classList.add('border-green-500');
            setTimeout(() => textarea.classList.remove('border-green-500'), 800);
          }
        } catch (err) {
          console.error('Speech2Text error:', err);
        }
      };

      mediaRecorder.start();
      recording = true;
      console.log('🎤 Recording...');
    } else {
      // Stop recording
      mediaRecorder.stop();
      recording = false;
      console.log('🛑 Recording stopped');
    }
  }

  // IP-Adapter Reference Image Upload
  function handleIpRefImageSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      ipRefImageFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        ipRefImagePreview = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  }

  async function uploadIpRefImage() {
    if (!ipRefImageFile) {
      warningMessage = 'Please select an image first';
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', ipRefImageFile);

      const response = await fetch('/api/ip_ref_image', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (data.status === 'success' || data.status === 'warning') {
        warningMessage = '';
        console.log('✅ IP-Adapter reference image uploaded successfully');
        console.log('Response:', data);
        // Show success message briefly
        const successMsg = document.createElement('div');
        successMsg.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 font-medium';
        successMsg.textContent = data.status === 'success' 
          ? '✅ Reference image uploaded successfully!' 
          : '⚠️ Image uploaded but may not be active';
        document.body.appendChild(successMsg);
        setTimeout(() => successMsg.remove(), 3000);
      } else {
        warningMessage = data.message || 'Failed to upload image';
      }
    } catch (err) {
      warningMessage = err instanceof Error ? err.message : 'Failed to upload image';
      console.error('IP-Adapter image upload error:', err);
    }
  }

  async function clearIpRefImage() {
    ipRefImageFile = null;
    ipRefImagePreview = null;
    const input = document.getElementById('ip-ref-image-input') as HTMLInputElement;
    if (input) input.value = '';
    
    // Clear backend IP-Adapter reference image
    try {
      const response = await fetch('/api/ip_ref_image/clear', {
        method: 'POST'
      });
      const data = await response.json();
      if (data.status === 'success') {
        console.log('✅ IP-Adapter reference image cleared');
        const successMsg = document.createElement('div');
        successMsg.className = 'fixed top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 font-medium';
        successMsg.textContent = '✅ Reference image cleared - IP-Adapter disabled';
        document.body.appendChild(successMsg);
        setTimeout(() => successMsg.remove(), 3000);
      }
    } catch (err) {
      console.error('Error clearing IP ref image:', err);
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <Warning bind:message={warningMessage}></Warning>
  <article class="text-center">
    {#if pageContent}
      {@html pageContent}
    {/if}
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {#if isImageMode}
        <div class="sm:col-start-1 flex flex-col gap-3">
          <VideoInput
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
          ></VideoInput>
          <!-- Generation Mode in the same column as VideoInput -->
          <div>
            <label for="mode" class="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2" style="font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;">
              Generation Mode:
            </label>
            <select
              id="mode"
              class="w-full border-2 border-gray-300 dark:border-gray-600 rounded-lg px-4 py-2.5 dark:text-black bg-white dark:bg-gray-100 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all font-medium"
              on:change={(e) => updateMode(e.target.value)}
            >
              <option value="full">Full Image</option>
              <option value="human">Human Only</option>
            </select>
          </div>
        </div>
      {/if}
      <div class={isImageMode ? 'sm:col-start-2' : 'col-span-2'}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-2">
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg my-1 p-2'}>
          {#if isLCMRunning}
            Stop
          {:else}
            Start
          {/if}
        </Button>

        <!-- speech input button -->
        <Button
          on:click={toggleRecording}
          classList={'text-lg my-1 p-2 bg-green-600 hover:bg-green-700'}
        >
          {recording ? '🛑 Stop Recording' : '🎤 Voice Input'}
        </Button>

        {#if recording}
          <div class="flex items-center gap-2 mt-1 text-red-500 animate-pulse">
            <div class="w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
            <p class="text-sm font-medium">Recording... speak now 🎙️</p>
          </div>
        {/if}

        <!-- Prompt input - moved above IP-Adapter Reference Image -->
        {#if pipelineParams.prompt}
          <div class="my-4">
            <TextArea params={pipelineParams.prompt} bind:value={$pipelineValues.prompt}></TextArea>
          </div>
        {/if}

        <!-- IP-Adapter Reference Image Upload with Scale inside -->
        <div class="my-4 p-5 border-2 border-blue-300 dark:border-blue-700 rounded-xl bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-gray-800 dark:via-gray-850 dark:to-gray-900 shadow-xl">
          <label class="block text-lg font-bold mb-3 text-gray-800 dark:text-gray-200" style="font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif; letter-spacing: -0.01em;">
            🎨 IP-Adapter Reference Image
          </label>
          <div class="flex flex-col gap-3">
            <div class="flex items-center gap-2 flex-wrap">
              <input
                id="ip-ref-image-input"
                type="file"
                accept="image/*"
                on:change={handleIpRefImageSelect}
                class="hidden"
              />
              <Button
                on:click={() => document.getElementById('ip-ref-image-input')?.click()}
                classList={'text-sm px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg shadow-md transition-all duration-200 font-medium'}
              >
                📷 Select Image
              </Button>
              {#if ipRefImageFile}
                <Button
                  on:click={uploadIpRefImage}
                  classList={'text-sm px-4 py-2 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white rounded-lg shadow-md transition-all duration-200 font-medium'}
                >
                  ⬆️ Upload
                </Button>
                <Button
                  on:click={clearIpRefImage}
                  classList={'text-sm px-4 py-2 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white rounded-lg shadow-md transition-all duration-200 font-medium'}
                >
                  ✕ Clear
                </Button>
              {/if}
            </div>
            {#if ipRefImagePreview}
              <div class="mt-3 p-3 bg-white dark:bg-gray-700 rounded-lg border-2 border-blue-300 dark:border-blue-600 shadow-inner">
                <p class="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">📸 Preview:</p>
                <div class="flex justify-center">
                  <img
                    src={ipRefImagePreview}
                    alt="IP-Adapter Reference"
                    class="max-w-full max-h-64 rounded-lg border-2 border-gray-300 dark:border-gray-600 object-contain shadow-md"
                  />
                </div>
              </div>
            {:else}
              <div class="mt-2 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 text-center">
                <p class="text-sm text-gray-500 dark:text-gray-400">No reference image selected</p>
              </div>
            {/if}
            <!-- IP-Adapter Scale slider inside the box -->
            {#if pipelineParams.ip_adapter_scale}
              <div class="mt-3 pt-3 border-t-2 border-blue-200 dark:border-blue-600">
                <InputRange params={pipelineParams.ip_adapter_scale} bind:value={$pipelineValues.ip_adapter_scale}></InputRange>
              </div>
            {/if}
          </div>
        </div>

        <!-- Other PipelineOptions (excluding prompt and ip_adapter_scale) -->
        {#if pipelineParams}
          {@const filteredParams = Object.fromEntries(
            Object.entries(pipelineParams).filter(([key, value]) => 
              key !== 'prompt' && key !== 'ip_adapter_scale'
            )
          )}
          {#if Object.keys(filteredParams).length > 0}
            <PipelineOptions pipelineParams={filteredParams}></PipelineOptions>
          {/if}
        {/if}

        <h1
          class="text-center mt-12 text-lg font-semibold text-black dark:text-white"
        >
          This Final Project is completed and authored by
          <span
            class="ml-2 font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-transparent bg-clip-text"
          >
          Ziqian Liu
          </span>
          and
          <span
            class="font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-transparent bg-clip-text"
          >
          Hongsheng Ye
          </span>
        </h1>

      </div>
    </article>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
