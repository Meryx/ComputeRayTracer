import React, { useEffect, useRef } from 'react';
import initGPUAndReturnDeviceAndContext from './newjs/GPU';
import createQuadProgram from './newjs/QuadProgram';
import createTextureAndReturnView from './newjs/Texture';
import createRaytraceProgram from './newjs/RayTracer';
import { WIDTH, HEIGHT, PRESENTATION_FORMAT } from './newjs/Constants';
import './App.css';

const renderLoop = ({ device, context, program, raytracer }) => {
  const frame = () => {
    /* Raytracing commands */
    const raytraceCommandEncoder = device.createCommandEncoder();
    const rayTracePass = raytraceCommandEncoder.beginComputePass();
    rayTracePass.setPipeline(raytracer.getPipeline());
    rayTracePass.setBindGroup(0, raytracer.getBindGroup());
    rayTracePass.dispatchWorkgroups(WIDTH, HEIGHT, 1);
    rayTracePass.end();
    const raytraceCommands = raytraceCommandEncoder.finish();
    device.queue.submit([raytraceCommands]);
    /* Raytracin END */

    /* Sample texture populated by raytracer */
    const texture = context.getCurrentTexture();
    const textureView = texture.createView();
    const renderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.5, g: 0.0, b: 0.25, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(program.getPipeline());
    passEncoder.setBindGroup(0, program.getBindGroup());
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    /* Rendering END */
  };
  requestAnimationFrame(frame);
};
const Main = () => {
  const hasMounted = useRef(false);
  useEffect(() => {
    const setup = async () => {
      const { device, context } = await initGPUAndReturnDeviceAndContext();
      const framebufferTextureDescriptor = {
        device,
        size: { width: WIDTH, height: HEIGHT },
        format: PRESENTATION_FORMAT,
        usageList: ['COPY_DST', 'STORAGE_BINDING', 'TEXTURE_BINDING'],
      };
      const framebuffer = createTextureAndReturnView(
        framebufferTextureDescriptor
      );
      const quadProgram = createQuadProgram({ device, framebuffer });
      const raytraceProgram = createRaytraceProgram({ device, framebuffer });
      renderLoop({
        device,
        context,
        program: quadProgram,
        raytracer: raytraceProgram,
      });
    };

    if (!hasMounted.current) {
      setup();
      hasMounted.current = true;
    }
  }, []);

  return (
    <div className="Main">
      <header className="App-header">
        <canvas id="canvas" width={WIDTH} height={HEIGHT}></canvas>
      </header>
    </div>
  );
};

export default Main;
