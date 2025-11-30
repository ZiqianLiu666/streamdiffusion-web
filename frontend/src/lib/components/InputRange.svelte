<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = 8.0;
  export let params: FieldProps;
  onMount(() => {
    value = Number(params?.default) ?? 8.0;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center gap-4 py-2">
  <label class="text-sm font-semibold text-gray-700 dark:text-gray-300" for={params.id} style="font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;">
    {params?.title}
  </label>
  <div class="col-span-2 relative flex items-center py-2">
    <input
      class="w-full cursor-pointer appearance-none"
      bind:value
      type="range"
      id={params.id}
      name={params.id}
      min={params?.min}
      max={params?.max}
      step={params?.step ?? 0.1}
      style="height: 12px;"
    />
  </div>
  <input
    type="number"
    step={params?.step ?? 0.1}
    bind:value
    class="rounded-lg border-2 border-gray-300 dark:border-gray-600 px-3 py-1.5 text-center text-sm font-bold dark:text-black bg-white dark:bg-gray-100 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all shadow-sm"
    style="min-width: 4rem;"
  />
</div>

<style>
  input[type='range'] {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    width: 100%;
  }
  
  input[type='range']::-webkit-slider-runnable-track {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right, #2563eb, #7c3aed, #db2777) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.2);
    border: 1.5px solid rgba(0, 0, 0, 0.2);
  }
  
  input[type='range']::-webkit-slider-track {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right, #2563eb, #7c3aed, #db2777) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.2);
    border: 1.5px solid rgba(0, 0, 0, 0.2);
  }
  
  input[type='range']::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    height: 26px;
    width: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6), 0 0 0 3px rgba(79, 70, 229, 0.3), inset 0 1px 2px rgba(255, 255, 255, 0.3);
    transition: all 0.2s ease;
    margin-top: -8px;
    border: 2.5px solid white;
  }
  
  input[type='range']::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.7), 0 0 0 4px rgba(79, 70, 229, 0.4), inset 0 1px 2px rgba(255, 255, 255, 0.4);
  }
  
  input[type='range']::-moz-range-track {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right, #2563eb, #7c3aed, #db2777) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.2);
    border: 1.5px solid rgba(0, 0, 0, 0.2);
  }
  
  input[type='range']::-moz-range-thumb {
    height: 26px;
    width: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    cursor: pointer;
    border: 2.5px solid white;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6), 0 0 0 3px rgba(79, 70, 229, 0.3);
    transition: all 0.2s ease;
  }
  
  input[type='range']::-moz-range-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.7), 0 0 0 4px rgba(79, 70, 229, 0.4);
  }
</style>
<!-- 
<style lang="postcss" scoped>
  input[type='range']::-webkit-slider-runnable-track {
    @apply h-2 cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-webkit-slider-thumb {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-moz-range-track {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-moz-range-thumb {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-ms-track {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-ms-thumb {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
</style> -->
