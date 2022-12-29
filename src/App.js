import React, { useEffect } from "react";
import redFragWGSL from "./shaders/red.frag.wgsl.json";
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
            code: `@vertex
            fn main(
              @builtin(vertex_index) VertexIndex : u32
            ) -> @builtin(position) vec4<f32> {
              var pos = array<vec2<f32>, 3>(
                vec2(0.0, 0.5),
                vec2(-0.5, -0.5),
                vec2(0.5, -0.5)
              );
            
              return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
            }`,
          }),
          entryPoint: "main",
        },
        fragment: {
          module: device.createShaderModule({
            code: redFragWGSL.code,
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
        // Sample is no longer the active page.

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

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

        device.queue.submit([commandEncoder.finish()]);
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
