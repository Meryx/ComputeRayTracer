import TextureRenderShaderCode from "./shaders/TextureRenderShader.wgsl";
import ComputeShaderCode from "./shaders/ComputeShader.wgsl";
import UpdateVariablesShaderCode from "./shaders/UpdateVariables.wgsl";
import cornell from "./scenes/cornell";
import CIE1931 from "./scenes/CIE.json";

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
      width: 700,
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

  scene.planarPatches = cornell.objects.patches.map((patch, i) => {
    return { ...patch, index: i };
  });
  let stride = 64;
  const planarPatchesBuffer = device.createBuffer({
    size: scene.planarPatches.length * stride,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const planarPatchesData = planarPatchesBuffer.getMappedRange();

  const obj = cornell.spectra;
  let array = [];
  const keyIndexPairs = {};

  Object.keys(obj).forEach((key, index) => {
    array.push(obj[key]);
    keyIndexPairs[key] = index;
  });

  const typeIndexPairs = {
    diffuse: 0,
    light: 1,
  };

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
    const emissionView = new Uint32Array(
      planarPatchesData,
      index * stride + 44,
      1
    );
    const spectrimIndexView = new Uint32Array(
      planarPatchesData,
      index * stride + 48,
      1
    );
    const typeView = new Uint32Array(planarPatchesData, index * stride + 52, 1);

    const indexView = new Uint32Array(
      planarPatchesData,
      index * stride + 56,
      1
    );

    originView.set(planarPatch.origin);
    edge1View.set(planarPatch.edge1);
    edge2View.set(planarPatch.edge2);
    emissionView.set([keyIndexPairs[planarPatch.emission]]);
    indexView.set([planarPatch.index]);
    spectrimIndexView.set([keyIndexPairs[planarPatch.reflectance]]);
    typeView.set([typeIndexPairs[planarPatch.type]]);
  });

  const planarPatchesBufferSize = scene.planarPatches.length * stride;
  planarPatchesBuffer.unmap();

  const lights = [];
  scene.planarPatches.forEach((planarPatch) => {
    if (planarPatch.type === "light") {
      lights.push(planarPatch);
    }
  });

  stride = 64;
  const lightsBuffer = device.createBuffer({
    size: lights.length * stride,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const lightsData = lightsBuffer.getMappedRange();

  lights.forEach((light, index) => {
    const originView = new Float32Array(lightsData, index * stride, 3);
    const edge1View = new Float32Array(lightsData, index * stride + 16, 3);
    const edge2View = new Float32Array(lightsData, index * stride + 32, 3);
    const emissionView = new Uint32Array(lightsData, index * stride + 44, 1);
    const spectrimIndexView = new Uint32Array(
      lightsData,
      index * stride + 48,
      1
    );
    const typeView = new Uint32Array(lightsData, index * stride + 52, 1);
    const indexView = new Uint32Array(lightsData, index * stride + 56, 1);

    originView.set(light.origin);
    edge1View.set(light.edge1);
    edge2View.set(light.edge2);
    emissionView.set([keyIndexPairs[light.emission]]);
    indexView.set([light.index]);
    spectrimIndexView.set([keyIndexPairs[light.reflectance]]);
    typeView.set([typeIndexPairs[light.type]]);
  });

  const lightsBufferSize = lightsData.byteLength;
  lightsBuffer.unmap();

  const altFramebufferData = new Float32Array(700 * 700 * 4);
  const altFramebuffer = device.createBuffer({
    size: altFramebufferData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const sample = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  new Uint32Array(sample.getMappedRange()).set([0]);
  sample.unmap();

  const camera = new Float32Array([
    ...cornell.camera.eye,
    0,
    ...cornell.camera.lookat,
    0,
    ...cornell.camera.up,
    cornell.camera.width,
    cornell.camera.height,
    cornell.camera.focalLength,
    0,
    0,
  ]);
  const cameraBuffer = device.createBuffer({
    size: camera.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  new Float32Array(cameraBuffer.getMappedRange()).set(camera);
  cameraBuffer.unmap();

  const lambdaMin = 400;
  const lambdaMax = 700;
  const range = lambdaMax - lambdaMin + 1;

  let narray = [];

  const sampleSpectrum = (spectrum, lambda) => {
    const index = spectrum.wavelength.findIndex((e) => e >= lambda);
    const start_index = Math.max(index - 1, 0);
    const end_index = Math.min(index, spectrum.wavelength.length - 1);
    const start = spectrum.value[start_index];
    const end = spectrum.value[end_index];
    const start_lambda = spectrum.wavelength[start_index];
    const end_lambda = spectrum.wavelength[end_index];
    if (start_lambda === end_lambda) return start;

    const val = lerp(
      start,
      end,
      (lambda - start_lambda) / (end_lambda - start_lambda)
    );
    return val;
  };

  for (let j = 0; j < array.length; j++) {
    for (let i = 0; i < range; i++) {
      const lambda = lambdaMin + i;
      const value = sampleSpectrum(array[j], lambda);
      narray.push(value);
    }
  }

  array = narray;
  const spectra = new Float32Array([...array]);
  const spectra_buffer = device.createBuffer({
    size: spectra.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const n = spectra_buffer.getMappedRange();
  const nview = new Float32Array(n);
  nview.set(spectra);

  spectra_buffer.unmap();

  const { CIE_X, CIE_Y, CIE_Z } = CIE1931;

  const CIE = new Float32Array([...CIE_X, ...CIE_Y, ...CIE_Z]);
  const CIE_buffer = device.createBuffer({
    size: CIE.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const m = CIE_buffer.getMappedRange();
  const dataview = new Float32Array(m);
  dataview.set(CIE);

  CIE_buffer.unmap();

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
          type: "storage",
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
          type: "read-only-storage",
        },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 7,
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
          buffer: altFramebuffer,
          size: altFramebufferData.byteLength,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: sample,
          size: 4,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: planarPatchesBuffer,
          size: planarPatchesBufferSize,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: cameraBuffer,
          size: camera.byteLength,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: CIE_buffer,
          size: CIE.byteLength,
        },
      },
      {
        binding: 6,
        resource: {
          buffer: spectra_buffer,
          size: spectra.byteLength,
        },
      },
      {
        binding: 7,
        resource: {
          buffer: lightsBuffer,
          size: lightsBufferSize,
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

  const updateVariablesPipelineBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const updateVariablesPipelineBindGroup = device.createBindGroup({
    layout: updateVariablesPipelineBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: sample,
          size: 4,
        },
      },
    ],
  });

  const updateVariablesPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [updateVariablesPipelineBindGroupLayout],
  });

  const updateVariablesPipeline = device.createComputePipeline({
    layout: updateVariablesPipelineLayout,
    compute: {
      module: device.createShaderModule({
        code: UpdateVariablesShaderCode,
      }),
      entryPoint: "main",
    },
  });

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
    let computePassEncoder = commandEncoder.beginComputePass();
    computePassEncoder.setPipeline(updateVariablesPipeline);
    computePassEncoder.setBindGroup(0, updateVariablesPipelineBindGroup);
    computePassEncoder.dispatchWorkgroups(1);
    computePassEncoder.end();
    computePassEncoder = commandEncoder.beginComputePass();
    computePassEncoder.setPipeline(computePipeline);
    computePassEncoder.setBindGroup(0, computePipelineBindGroup);
    computePassEncoder.dispatchWorkgroups(
      Math.ceil(700 / 8),
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
    device.queue.submit([commands]);
    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);
};

const lerp = (a, b, t) => a * (1 - t) + b * t;

export { Main };
