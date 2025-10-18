declare module "*.svelte?async-render" {
  import type { ComponentType } from "svelte";

  const component: ComponentType;
  export default component;
}
