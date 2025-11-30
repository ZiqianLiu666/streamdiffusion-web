<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';
  import { onMount, onDestroy } from 'svelte';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  
  // FPS calculation
  let fps: number = 0;
  let displayFps: number = 0;
  const frameTimes: number[] = [];
  const maxFrameHistory = 30; // Keep last 30 frames for averaging
  let fpsUpdateInterval: ReturnType<typeof setInterval> | null = null;
  let frameCheckInterval: ReturnType<typeof setInterval> | null = null;
  let rafId: number | null = null;
  let lastCheckTime: number = 0;
  let lastImageData: string = '';
  
  function calculateFPS() {
    const now = performance.now();
    frameTimes.push(now);
    
    // Keep only recent frames
    if (frameTimes.length > maxFrameHistory) {
      frameTimes.shift();
    }
    
    // Calculate FPS from frame intervals
    if (frameTimes.length >= 2) {
      const timeSpan = frameTimes[frameTimes.length - 1] - frameTimes[0];
      const frameCount = frameTimes.length - 1;
      fps = (frameCount / timeSpan) * 1000;
    }
  }
  
  function onImageLoad() {
    if (isLCMRunning && imageEl) {
      // For MJPEG streams, onload may not fire for every frame
      // But when it does, we definitely have a new frame
      calculateFPS();
    }
  }
  
  // FPS detection for MJPEG streams - use a more reliable method
  let lastFrameTime: number = 0;
  let frameCount: number = 0;
  let fpsStartTime: number = 0;
  
  function detectFrameUpdate() {
    if (!isLCMRunning || !imageEl) return;
    
    const now = performance.now();
    
    // Check if image is valid and loaded
    if (imageEl.complete && imageEl.naturalWidth > 0) {
      // For MJPEG streams, we use time-based detection
      // Since the stream updates continuously, we count frames over time
      const elapsed = now - lastFrameTime;
      if (elapsed > 16) { // At least 16ms between frames (60fps max)
        frameCount++;
        if (fpsStartTime === 0) {
          fpsStartTime = now;
        }
        
        // Calculate FPS every second
        const timeSpan = now - fpsStartTime;
        if (timeSpan >= 1000) {
          fps = (frameCount / timeSpan) * 1000;
          frameCount = 0;
          fpsStartTime = now;
        }
        
        lastFrameTime = now;
      }
    }
  }
  
  function onImageError() {
    // Reset FPS on error
    if (!isLCMRunning) {
      stopFpsUpdate();
    }
  }
  
  function startFpsUpdate() {
    if (fpsUpdateInterval) return;
    
    // Update display FPS smoothly
    fpsUpdateInterval = setInterval(() => {
      // Smooth FPS display with exponential moving average
      if (fps > 0) {
        displayFps = displayFps * 0.7 + fps * 0.3;
      } else if (displayFps > 0) {
        // Gradually decrease if no new frames
        displayFps = displayFps * 0.9;
      }
    }, 100);
    
    // Check for frame updates using requestAnimationFrame for smooth detection
    const checkLoop = () => {
      if (isLCMRunning) {
        detectFrameUpdate();
        rafId = requestAnimationFrame(checkLoop);
      } else {
        rafId = null;
      }
    };
    rafId = requestAnimationFrame(checkLoop);
    
    // Also use interval for MJPEG frame detection (more aggressive)
    frameCheckInterval = setInterval(() => {
      if (isLCMRunning) {
        detectFrameUpdate();
      }
    }, 33); // Check every 33ms (~30fps detection rate)
  }
  
  function stopFpsUpdate() {
    if (fpsUpdateInterval) {
      clearInterval(fpsUpdateInterval);
      fpsUpdateInterval = null;
    }
    if (frameCheckInterval) {
      clearInterval(frameCheckInterval);
      frameCheckInterval = null;
    }
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    fps = 0;
    displayFps = 0;
    frameTimes.length = 0;
    lastImageData = '';
    lastCheckTime = 0;
    lastFrameTime = 0;
    frameCount = 0;
    fpsStartTime = 0;
  }
  
  $: if (isLCMRunning) {
    startFpsUpdate();
  } else {
    stopFpsUpdate();
  }
  
  onDestroy(() => {
    stopFpsUpdate();
  });
  
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }
</script>

<div
  class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="aspect-square w-full rounded-lg"
      src={'/api/stream/' + $streamId}
      on:load={onImageLoad}
      on:error={onImageError}
    />
    <!-- FPS Display - Always visible when stream is running -->
    <div class="absolute top-4 left-4 z-50 bg-gradient-to-br from-black/90 via-black/85 to-black/80 backdrop-blur-md rounded-xl px-5 py-3 shadow-2xl border-2 border-white/30">
      <div class="flex items-baseline gap-3">
        <span class="text-[10px] font-medium text-gray-300 uppercase tracking-[0.15em] leading-none">FPS</span>
        <span 
          class="text-3xl font-bold text-white tabular-nums select-none" 
          style="font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', 'Consolas', monospace; letter-spacing: -0.03em; text-shadow: 0 2px 10px rgba(0,0,0,0.8), 0 0 20px rgba(59, 130, 246, 0.5); min-width: 3.5rem;"
        >
          {isLCMRunning ? (displayFps > 0 ? displayFps.toFixed(1) : '0.0') : '0.0'}
        </span>
      </div>
      <div class="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-8 h-0.5 bg-gradient-to-r from-transparent via-blue-400/50 to-transparent"></div>
    </div>
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>
