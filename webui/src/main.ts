import "./app.css";
import App from "./App.svelte";
import { mount } from "svelte";

const target = document.getElementById("app");

if (!target) {
  throw new Error("Failed to mount Svelte application");
}

const app = mount(App, {
  target,
});

export default app;
