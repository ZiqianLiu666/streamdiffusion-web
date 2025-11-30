<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value: string;
  export let params: FieldProps;
  onMount(() => {
    value = String(params?.default ?? '');
  });
</script>

<div class="py-2">
  <label class="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2" for={params?.title} style="font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;">
    {params?.title}
  </label>
  <div class="text-normal flex items-center rounded-lg border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-100 shadow-sm focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-200 transition-all">
    <textarea
      class="mx-1 w-full px-4 py-3 font-normal text-gray-800 dark:text-black outline-none resize-none rounded-lg"
      title={params?.title}
      placeholder="Add your prompt here..."
      bind:value
      rows="3"
    ></textarea>
  </div>
</div>
