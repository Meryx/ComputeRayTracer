import TextureRenderShaderCode from "./shaders/TextureRenderShader.wgsl";
import ComputeShaderCode from "./shaders/ComputeShader.wgsl";
import { vec3, vec4 } from "gl-matrix";

class Renderer {
  constructor({ device, context, integrator, scene, sample }) {
    this.device = device;
    this.context = context;
    this.integrator = integrator;
    this.scene = scene;
    this.sample = sample;
  }

  render({
    renderPipeline,
    renderPipelineBindGroup,
    computePipeline,
    computePipelineBindGroup,
  }) {
    const { device, context, integrator, scene, sample } = this;
    const { queue } = device;
    let s = 1;
    queue.writeBuffer(sample, 0, new Uint32Array([1]));
    const frame = () => {
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
      const commandEncoder = device.createCommandEncoder();
      const computePassEncoder = commandEncoder.beginComputePass();
      computePassEncoder.setPipeline(computePipeline);
      computePassEncoder.setBindGroup(0, computePipelineBindGroup);
      computePassEncoder.dispatchWorkgroups(
        Math.ceil(1400 / 8),
        Math.ceil(700 / 8),
        1
      );
      computePassEncoder.end();
      const renderPassEncoder =
        commandEncoder.beginRenderPass(renderPassDescriptor);
      renderPassEncoder.setPipeline(renderPipeline);
      renderPassEncoder.setBindGroup(0, renderPipelineBindGroup);
      renderPassEncoder.draw(6, 1, 0, 0);
      renderPassEncoder.end();
      const commands = commandEncoder.finish();
      queue.submit([commands]);
      requestAnimationFrame(frame);
      s = s + 1;
      queue.writeBuffer(sample, 0, new Uint32Array([s]));
    };

    requestAnimationFrame(frame);
  }
}

const Main = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  device.addEventListener("uncapturederror", (event) => {
    // Re-surface the error, because adding an event listener may silence console logs.
    console.error("A WebGPU error was not captured:", event.error);
  });

  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("webgpu");

  const format = "rgba8unorm";
  context.configure({
    device,
    format,
  });

  const framebufferDescriptor = {
    size: {
      width: 1400,
      height: 700,
    },
    format,
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING,
  };

  const framebuffer = device.createTexture(framebufferDescriptor);
  const framebufferView = framebuffer.createView();

  const framebufferSamplerDescriptor = {
    magFilter: "linear",
  };
  const framebufferSampler = device.createSampler(framebufferSamplerDescriptor);
  const textureRenderShaderModule = device.createShaderModule({
    code: TextureRenderShaderCode,
  });

  const vertexStage = {
    module: textureRenderShaderModule,
    entryPoint: "vert_main",
  };

  const fragmentStage = {
    module: textureRenderShaderModule,
    entryPoint: "frag_main",
    targets: [{ format }],
  };

  const renderPipelinBindGroupLayoutDescriptor = {
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
    ],
  };

  const renderPipelineBingGroupLayout = device.createBindGroupLayout(
    renderPipelinBindGroupLayoutDescriptor
  );

  const renderPipelinBindGroupDescriptor = {
    layout: renderPipelineBingGroupLayout,
    entries: [
      {
        binding: 0,
        resource: framebufferSampler,
      },
      {
        binding: 1,
        resource: framebufferView,
      },
    ],
  };

  const renderPipelineBindGroup = device.createBindGroup(
    renderPipelinBindGroupDescriptor
  );

  const renderPipelineLayoutDescriptor = {
    bindGroupLayouts: [renderPipelineBingGroupLayout],
  };

  const renderPipelineLayout = device.createPipelineLayout(
    renderPipelineLayoutDescriptor
  );

  const renderPipelineDescriptor = {
    layout: renderPipelineLayout,
    vertex: vertexStage,
    fragment: fragmentStage,
  };

  const renderPipeline = device.createRenderPipeline(renderPipelineDescriptor);

  const scene = {};

  const redSphere = {
    geometry: vec4.fromValues(0.0, -3.0, 978, 10),
    index: 0,
  };

  const groundSphere = {
    geometry: vec4.fromValues(0.0, -100.5, -1.0, 100.0),
    index: 1,
  };

  scene.objects = [redSphere];

  let stride = 32;

  const { objects } = scene;
  const objectData = new ArrayBuffer(objects.length * stride);
  objects.forEach((object, index) => {
    const geometryView = new Float32Array(objectData, index * stride, 4);
    const indexView = new Uint32Array(objectData, index * stride + 16, 1);
    geometryView.set(object.geometry);
    indexView.set([object.index]);
  });
  const objectBuffer = device.createBuffer({
    size: objectData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bluePlanarPatch = {
    origin: vec3.fromValues(-800, -1000, 700.0),
    edge1: vec3.fromValues(0, 0.0, -1200),
    edge2: vec3.fromValues(0.0, 2200, 0.0),
    index: 2,
  };

  const rightPlanarPatch = {
    origin: vec3.fromValues(800, -1000, 700.0),
    edge1: vec3.fromValues(0.0, 2200, 0.0),
    edge2: vec3.fromValues(0, 0.0, -1200),
    index: 3,
  };

  const topPlanarPatch = {
    origin: vec3.fromValues(-1200, 1000, 700.0),
    edge2: vec3.fromValues(2400, 0.0, 0),
    edge1: vec3.fromValues(0.0, 0, -1800.0),
    index: 4,
  };

  const backPlanarPatch = {
    origin: vec3.fromValues(-1200, -1000, -500.0),
    edge2: vec3.fromValues(0.0, 2000, 0),
    edge1: vec3.fromValues(2400, 0.0, 0),
    index: 5,
  };

  const bottomPlanarPatch = {
    origin: vec3.fromValues(-1200, -1500, 500.0),
    edge1: vec3.fromValues(2400, 0.0, 0),
    edge2: vec3.fromValues(0.0, 0, -1800.0),
    index: 6,
  };

  scene.planarPatches = [
    topPlanarPatch,
    bluePlanarPatch,
    rightPlanarPatch,
    backPlanarPatch,
    bottomPlanarPatch,
  ];
  stride = 48;
  const planarPatchesData = new ArrayBuffer(
    scene.planarPatches.length * stride
  );

  scene.planarPatches.forEach((planarPatch, index) => {
    const originView = new Float32Array(planarPatchesData, index * stride, 3);
    const edge1View = new Float32Array(
      planarPatchesData,
      index * stride + 16,
      3
    );
    const edge2View = new Float32Array(
      planarPatchesData,
      index * stride + 32,
      3
    );
    const indexView = new Uint32Array(
      planarPatchesData,
      index * stride + 44,
      1
    );
    originView.set(planarPatch.origin);
    edge1View.set(planarPatch.edge1);
    edge2View.set(planarPatch.edge2);
    indexView.set([planarPatch.index]);
  });

  const planarPatchesBuffer = device.createBuffer({
    size: planarPatchesData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const altFramebufferData = new Float32Array(1400 * 700 * 4);
  const altFramebuffer = device.createBuffer({
    size: altFramebufferData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const sample = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const computePipelineBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: "write-only",
          format,
          viewDimension: "2d",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
    ],
  });

  const computePipelineBindGroup = device.createBindGroup({
    layout: computePipelineBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: framebufferView,
      },
      {
        binding: 1,
        resource: {
          buffer: objectBuffer,
          size: objectData.byteLength,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: altFramebuffer,
          size: altFramebufferData.byteLength,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: sample,
          size: 4,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: planarPatchesBuffer,
          size: planarPatchesData.byteLength,
        },
      },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computePipelineBindGroupLayout],
  });

  const computePipeline = device.createComputePipeline({
    layout: computePipelineLayout,
    compute: {
      module: device.createShaderModule({
        code: ComputeShaderCode,
      }),
      entryPoint: "main",
    },
  });

  device.queue.writeBuffer(objectBuffer, 0, objectData);
  device.queue.writeBuffer(planarPatchesBuffer, 0, planarPatchesData);

  const renderer = new Renderer({ device, context, scene, sample });
  renderer.render({
    renderPipeline,
    renderPipelineBindGroup,
    computePipeline,
    computePipelineBindGroup,
  });
};

export default Renderer;
export { Main };
