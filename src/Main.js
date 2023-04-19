import React, { useEffect, useRef } from 'react';
import initGPUAndReturnDeviceAndContext from './newjs/GPU';
import createQuadProgram from './newjs/QuadProgram';
import createTextureAndReturnView from './newjs/Texture';
import {
  createRaytraceProgram,
  loadMeshIntoRayTraceProgram,
  loadTextureIntoRayTraceProgram,
  incrementSample,
} from './newjs/RayTracer';
import createBuffer from './newjs/Buffer';
import { WIDTH, HEIGHT, PRESENTATION_FORMAT } from './newjs/Constants';
import './Main.css';
import './App.css';

const createIndexedMesh = (objData) => {
  const lines = objData.split('\n');
  const vertices = [];
  const triangles = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('v ')) {
      const values = line.split(/\s+/);
      const x = parseFloat(values[1]);
      const y = parseFloat(values[2]);
      const z = parseFloat(values[3]);
      vertices.push([x, y, z]);
    } else if (line.startsWith('f ')) {
      const values = line.split(/\s+/);
      const v1 = parseInt(values[1]) - 1;
      const v2 = parseInt(values[2]) - 1;
      const v3 = parseInt(values[3]) - 1;
      triangles.push([v1, v2, v3]);
    }
  }
  return { triangles, vertices };
};

const loadMesh = (e) => {
  e.preventDefault();
  const input = document.getElementById('mesh-upload-button');
  const file = input.files[0];
  const reader = new FileReader();
  reader.readAsText(file);
  reader.onload = (event) => {
    const objData = event.target.result;
    const { triangles, vertices } = createIndexedMesh(objData);

    const triangleArray = Array.from(
      triangles,
      ([v1, v2, v3]) => new Uint32Array([v1, v2, v3, 0])
    );

    /* Load triangles */
    const triangleBufferArray = triangleArray.reduce((buffer, array) => {
      const newBuffer = new Uint32Array(
        buffer.byteLength / Uint32Array.BYTES_PER_ELEMENT + array.length
      );
      newBuffer.set(new Uint32Array(buffer));
      newBuffer.set(
        new Uint32Array(array.buffer),
        buffer.byteLength / Uint32Array.BYTES_PER_ELEMENT
      );
      return newBuffer.buffer;
    }, new ArrayBuffer(0));

    /* Add triangle list to shader */
    const triangleBufferDescriptor = {
      device,
      size: triangleBufferArray.byteLength,
      usageList: ['COPY_DST', 'STORAGE'],
    };
    const triangleBuffer = createBuffer(triangleBufferDescriptor);

    /* Add vertex list to shader */
    const vertexArray = Array.from(
      vertices,
      ([v1, v2, v3]) => new Float32Array([v1, v2, v3, 0])
    );

    const vertexBufferArray = vertexArray.reduce((buffer, array) => {
      const newBuffer = new Float32Array(
        buffer.byteLength / Float32Array.BYTES_PER_ELEMENT + array.length
      );
      newBuffer.set(new Float32Array(buffer));
      newBuffer.set(
        new Float32Array(array.buffer),
        buffer.byteLength / Float32Array.BYTES_PER_ELEMENT
      );
      return newBuffer.buffer;
    }, new ArrayBuffer(0));
    const vertexBufferDescriptor = {
      device,
      size: vertexBufferArray.byteLength,
      usageList: ['COPY_DST', 'STORAGE'],
    };
    const vertexBuffer = createBuffer(vertexBufferDescriptor);

    loadMeshIntoRayTraceProgram({
      triangleBuffer,
      vertexBuffer,
      numOfTriangles: triangleBufferArray.byteLength / 16,
    });
    device.queue.writeBuffer(triangleBuffer, 0, triangleBufferArray);
    device.queue.writeBuffer(vertexBuffer, 0, vertexBufferArray);
    requestAnimationFrame(() => renderLoop({ device, context }));
  };
};

const loadTexture = (e) => {
  const canvas = document.getElementById('hiddencanvas');
  const ctx = canvas.getContext('2d');
  e.preventDefault();
  const input = document.getElementById('texture-upload-button');
  const file = input.files[0];
  const imageUrl = URL.createObjectURL(file);
  const image = new Image();
  image.src = imageUrl;
  image.onload = function () {
    // Set the canvas size to match the image size
    canvas.width = image.width;
    canvas.height = image.height;

    // Draw the image onto the canvas
    ctx.drawImage(image, 0, 0);

    // Extract the image data (RGBA format) from the canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imageData;
    loadTextureIntoRayTraceProgram({ data, width, height });
    requestAnimationFrame(() => renderLoop({ device, context }));
  };

  // const reader = new FileReader();
  // reader.readAsBinaryString(file);
  // reader.onload = (event) => {
  //   const f = event.target.result;
  //   const image = decodeJpg({ data: f });
  //   console.log(image);
  // };
};

const renderLoop = ({ device, context }) => {
  let rendered = false;
  const frame = async () => {
    if (true) {
      /* Raytracing commands */
      const raytraceCommandEncoder = device.createCommandEncoder();
      const rayTracePass = raytraceCommandEncoder.beginComputePass();
      rayTracePass.setPipeline(raytraceProgram.getPipeline());
      rayTracePass.setBindGroup(0, raytraceProgram.getBindGroup());
      rayTracePass.dispatchWorkgroups(WIDTH, HEIGHT, 1);
      rayTracePass.end();
      const raytraceCommands = raytraceCommandEncoder.finish();
      device.queue.submit([raytraceCommands]);
      rendered = true;
      incrementSample();

      /* Raytracing END */
    }

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
    passEncoder.setPipeline(quadProgram.getPipeline());
    passEncoder.setBindGroup(0, quadProgram.getBindGroup());
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    /* Rendering END */
    await device.queue.onSubmittedWorkDone();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
};

let raytraceProgram;
let quadProgram;
let device;
let context;
let textureArrayBuffer;

const Main = () => {
  const hasMounted = useRef(false);
  useEffect(() => {
    const setup = async () => {
      ({ device, context } = await initGPUAndReturnDeviceAndContext());
      const framebufferTextureDescriptor = {
        device,
        size: { width: WIDTH, height: HEIGHT },
        format: PRESENTATION_FORMAT,
        usageList: ['COPY_DST', 'STORAGE_BINDING', 'TEXTURE_BINDING'],
      };
      const framebuffer = createTextureAndReturnView(
        framebufferTextureDescriptor
      );
      quadProgram = createQuadProgram({ device, framebuffer });
      raytraceProgram = createRaytraceProgram({ device, framebuffer });
      renderLoop({
        device,
        context,
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
        <canvas id="hiddencanvas" style={{ display: 'none' }}></canvas>
        <canvas id="canvas" width={WIDTH} height={HEIGHT}></canvas>
        <div className="container">
          <div>
            <form>
              <label htmlFor="mesh-upload-button">Upload Mesh</label>
              <input
                type="file"
                id="mesh-upload-button"
                name="file"
                accept=".obj"
              />
              <br />
              <button type="submit" onClick={loadMesh}>
                Submit
              </button>
            </form>
          </div>
          <div>
            <form>
              <label htmlFor="texture-upload-button">Upload Texture</label>
              <input
                type="file"
                id="texture-upload-button"
                name="file"
                accept=".png,.jpg,.jpeg,.bmp"
              />
              <br />
              <button type="submit" onClick={loadTexture}>
                Submit
              </button>
            </form>
          </div>
        </div>
      </header>
    </div>
  );
};

export default Main;
