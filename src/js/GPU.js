import computeShader from "../shaders/compute.wgsl";
import renderShader from "../shaders/screen_shader.wgsl";

const init = async (WIDTH, HEIGHT) => {
  //Retrieve GPU interface
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("webgpu");

  // RGBA order, 8 bits, unsigned, normalized
  const presentationFormat = "rgba8unorm";
  // const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "opaque",
  });

  const color_buffer = device.createTexture({
    size: {
      width: WIDTH,
      height: HEIGHT,
    },
    format: presentationFormat,
    usage:
      window.GPUTextureUsage.COPY_DST |
      window.GPUTextureUsage.STORAGE_BINDING |
      window.GPUTextureUsage.TEXTURE_BINDING,
  });

  const color_buffer_view = color_buffer.createView();

  const sampler = device.createSampler({
    addressModeU: "repeat",
    addressModeV: "repeat",
    magFilter: "linear",
    minFilter: "nearest",
    mipmapFilter: "nearest",
    maxAnisotropy: 1,
  });

  const cameraBufferSize = 11 * Float32Array.BYTES_PER_ELEMENT;

  const cameraBuffer = device.createBuffer({
    size: cameraBufferSize,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const viewMatSize = 16 * Float32Array.BYTES_PER_ELEMENT;

  const viewMatBuffer = device.createBuffer({
    size: viewMatSize,
    usage: window.GPUBufferUsage.COPY_DST | window.GPUBufferUsage.UNIFORM,
  });

  const worldBuffer = device.createBuffer({
    size: 11 * Float32Array.BYTES_PER_ELEMENT,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const rayTracingBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: window.GPUShaderStage.COMPUTE,
        storageTexture: {
          access: "write-only",
          format: presentationFormat,
          viewDimension: "2d",
        },
      },
      {
        binding: 1,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
          hasDynamicOffset: false,
          minBindingSize: 0,
        },
      },
      {
        binding: 2,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {},
      },
      {
        binding: 3,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
          hasDynamicOffset: false,
          minBindingSize: 0,
        },
      },
    ],
  });

  const rayTracingBindGroup = device.createBindGroup({
    layout: rayTracingBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: color_buffer_view,
      },
      {
        binding: 1,
        resource: {
          buffer: cameraBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: viewMatBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: worldBuffer,
        },
      },
    ],
  });

  const rayTracingPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [rayTracingBindGroupLayout],
  });

  const rayTracingPipeline = device.createComputePipeline({
    layout: rayTracingPipelineLayout,
    compute: {
      module: device.createShaderModule({
        code: computeShader,
      }),
      entryPoint: "main",
    },
  });

  const screenBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: window.GPUShaderStage.FRAGMENT,
        sampler: {},
      },
      {
        binding: 1,
        visibility: window.GPUShaderStage.FRAGMENT,
        texture: {},
      },
    ],
  });

  const screenBindGroup = device.createBindGroup({
    layout: screenBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: sampler,
      },
      {
        binding: 1,
        resource: color_buffer_view,
      },
    ],
  });

  const screenPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [screenBindGroupLayout],
  });

  const screenPipeline = device.createRenderPipeline({
    layout: screenPipelineLayout,
    vertex: {
      module: device.createShaderModule({
        code: renderShader,
      }),
      entryPoint: "vert_main",
    },
    fragment: {
      module: device.createShaderModule({
        code: renderShader,
      }),
      entryPoint: "frag_main",
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

  const frame = () => {
    const commandEncoder = device.createCommandEncoder();
    const rayTracePass = commandEncoder.beginComputePass();
    rayTracePass.setPipeline(rayTracingPipeline);
    rayTracePass.setBindGroup(0, rayTracingBindGroup);
    rayTracePass.dispatchWorkgroups(WIDTH, HEIGHT, 1);
    rayTracePass.end();

    const texture = context.getCurrentTexture();
    const textureView = texture.createView();
    const renderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.5, g: 0.0, b: 0.25, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(screenPipeline);
    passEncoder.setBindGroup(0, screenBindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);

  return { device, cameraBuffer, viewMatBuffer, worldBuffer };
};

const functions = { init };

export default functions;
