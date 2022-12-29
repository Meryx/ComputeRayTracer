import React, { useEffect } from "react";
import redFragWGSL from "./shaders/red.frag.wgsl";
import triangleVertWGSL from "./shaders/triangle.vert.wgsl";
import "./App.css";

function App() {
  useEffect(() => {
    const setup = async () => {
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const canvas = document.getElementById("canvas");
      const context = canvas.getContext("webgpu");
      const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device,
        format: presentationFormat,
        alphaMode: "opaque",
      });
      const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: device.createShaderModule({
            code: triangleVertWGSL,
          }),
          entryPoint: "main",
        },
        fragment: {
          module: device.createShaderModule({
            code: redFragWGSL,
          }),
          entryPoint: "main",
          targets: [
            {
              format: presentationFormat,
            },
          ],
        },
        primitive: {
          topology: "triangle-list",
        },
      });

      function frame() {
        const commandEncoder = device.createCommandEncoder();
        const texture = context.getCurrentTexture();
        const textureView = texture.createView();
        const renderPassDescriptor = {
          colorAttachments: [
            {
              view: textureView,
              clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        };

        const passEncoder =
          commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.draw(3, 1, 0, 0);
        passEncoder.end();
        const commands = commandEncoder.finish();
        device.queue.submit([commands]);
        requestAnimationFrame(frame);
      }

      requestAnimationFrame(frame);
    };

    setup();
  });

  return (
    <div className="App">
      <header className="App-header">
        <canvas id="canvas" width="200" height="200"></canvas>
      </header>
    </div>
  );
}

export default App;
