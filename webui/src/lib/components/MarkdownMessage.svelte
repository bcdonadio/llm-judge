<script lang="ts">
  import { marked } from "marked";
  import DOMPurify from "isomorphic-dompurify";

  interface Props {
    content: string;
  }

  let { content }: Props = $props();

  // Configure marked options
  marked.setOptions({
    breaks: true, // Convert \n to <br>
    gfm: true, // GitHub Flavored Markdown
  });

  // Parse markdown and sanitize HTML
  const htmlContent = $derived(() => {
    if (!content) return "";

    // Convert markdown to HTML
    const markdown = marked(content) as string;

    // Add security hook to prevent reverse tabnabbing attacks
    DOMPurify.addHook("afterSanitizeAttributes", (node) => {
      // Add rel='noopener noreferrer' to links with target='_blank'
      if (node.tagName === "A" && node.getAttribute("target") === "_blank") {
        node.setAttribute("rel", "noopener noreferrer");
      }
    });

    // Sanitize HTML to prevent XSS attacks
    const sanitized = DOMPurify.sanitize(markdown, {
      ALLOWED_TAGS: [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "br",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "s",
        "a",
        "ul",
        "ol",
        "li",
        "blockquote",
        "code",
        "pre",
        "hr",
      ],
      ALLOWED_ATTR: ["href", "title", "target", "rel"],
    });

    // Remove the hook after sanitization to avoid memory leaks
    DOMPurify.removeHook("afterSanitizeAttributes");

    return sanitized;
  });
</script>

<div class="markdown-content">
  <!-- eslint-disable-next-line svelte/no-at-html-tags -->
  {@html htmlContent()}
</div>

<style>
  .markdown-content {
    /* Inherit text color and font from parent */
    color: inherit;
    font-family: inherit;
    line-height: 1.6;
  }

  /* Headings */
  .markdown-content :global(h1) {
    font-size: 1.5em;
    font-weight: 600;
    margin: 0.3em 0 0.2em 0;
  }

  .markdown-content :global(h2) {
    font-size: 1.3em;
    font-weight: 600;
    margin: 0.3em 0 0.2em 0;
  }

  .markdown-content :global(h3) {
    font-size: 1.15em;
    font-weight: 600;
    margin: 0.3em 0 0.2em 0;
  }

  .markdown-content :global(h4),
  .markdown-content :global(h5),
  .markdown-content :global(h6) {
    font-size: 1em;
    font-weight: 600;
    margin: 0.3em 0 0.2em 0;
  }

  /* Paragraphs */
  .markdown-content :global(p) {
    margin: 0;
  }

  /* First and last elements shouldn't have extra margin */
  .markdown-content :global(> :first-child) {
    margin-top: 0;
  }

  .markdown-content :global(> :last-child) {
    margin-bottom: 0;
  }

  /* Lists */
  .markdown-content :global(ul) {
    margin: 0.3em 0;
    padding-left: 1.5em;
    list-style-type: disc;
  }

  .markdown-content :global(ol) {
    margin: 0.3em 0;
    padding-left: 1.5em;
    list-style-type: decimal;
  }

  .markdown-content :global(li) {
    margin: 0;
    display: list-item;
  }

  /* Links */
  .markdown-content :global(a) {
    color: var(--color-accent, #5fd3bc);
    text-decoration: underline;
    text-decoration-thickness: 1px;
    text-underline-offset: 2px;
  }

  .markdown-content :global(a:hover) {
    text-decoration-thickness: 2px;
  }

  /* Code */
  .markdown-content :global(code) {
    background: rgba(255, 255, 255, 0.05);
    padding: 0.1em 0.3em;
    border-radius: 3px;
    font-family: var(--font-mono, "Courier New", monospace);
    font-size: 0.9em;
  }

  .markdown-content :global(pre) {
    background: rgba(255, 255, 255, 0.05);
    padding: 0.8em;
    border-radius: 4px;
    overflow-x: auto;
    margin: 0.3em 0;
  }

  .markdown-content :global(pre code) {
    background: none;
    padding: 0;
  }

  /* Blockquotes */
  .markdown-content :global(blockquote) {
    border-left: 3px solid var(--color-accent, #5fd3bc);
    padding-left: 1em;
    margin: 0.3em 0;
    opacity: 0.9;
  }

  /* Emphasis */
  .markdown-content :global(strong),
  .markdown-content :global(b) {
    font-weight: 600;
  }

  .markdown-content :global(em),
  .markdown-content :global(i) {
    font-style: italic;
  }

  /* Horizontal rule */
  .markdown-content :global(hr) {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin: 0.5em 0;
  }
</style>
