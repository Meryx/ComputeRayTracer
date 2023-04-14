import computeShader from '../shaders/compute.wgsl';
import renderShader from '../shaders/screen_shader.wgsl';

const TILE_WIDTH = 32;
const TILE_HEIGHT = 32;

let currentTileX = 0;
let currentTileY = 0;

const init = async (WIDTH, HEIGHT) => {
  //Retrieve GPU interface
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');

  // RGBA order, 8 bits, unsigned, normalized
  const presentationFormat = 'rgba8unorm';
  // const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'opaque',
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
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'nearest',
    mipmapFilter: 'nearest',
    maxAnisotropy: 1,
  });

  const cameraBufferSize = 11 * Float32Array.BYTES_PER_ELEMENT;

  const cameraBuffer = device.createBuffer({
    size: cameraBufferSize,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const createdSize = 1 * Uint32Array.BYTES_PER_ELEMENT;

  const createdBuffer = device.createBuffer({
    size: createdSize,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const tileSize = 3 * Uint32Array.BYTES_PER_ELEMENT;

  const tileBuffer = device.createBuffer({
    size: tileSize,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const viewMatSize = 16 * Float32Array.BYTES_PER_ELEMENT;

  const viewMatBuffer = device.createBuffer({
    size: viewMatSize,
    usage: window.GPUBufferUsage.COPY_DST | window.GPUBufferUsage.UNIFORM,
  });

  const worldBuffer = device.createBuffer({
    size: 500 * 96,
    usage: window.GPUBufferUsage.UNIFORM | window.GPUBufferUsage.COPY_DST,
  });

  const treeBuffer = device.createBuffer({
    size: 550 * 48,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const alt_color_buffer = device.createBuffer({
    size: 960000 * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: window.GPUBufferUsage.STORAGE | window.GPUBufferUsage.COPY_DST,
  });

  const rayTracingBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: window.GPUShaderStage.COMPUTE,
        storageTexture: {
          access: 'write-only',
          format: presentationFormat,
          viewDimension: '2d',
        },
      },
      {
        binding: 1,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
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
          hasDynamicOffset: false,
          minBindingSize: 0,
        },
      },
      {
        binding: 4,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
          hasDynamicOffset: false,
          minBindingSize: 0,
        },
      },
      {
        binding: 5,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
          hasDynamicOffset: false,
          minBindingSize: 0,
        },
      },
      {
        binding: 6,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: 'storage',
          hasDynamicOffset: false,
          minBindingSize: 0,
        },
      },
      {
        binding: 7,
        visibility: window.GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
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
      {
        binding: 4,
        resource: {
          buffer: createdBuffer,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: tileBuffer,
        },
      },
      {
        binding: 6,
        resource: {
          buffer: alt_color_buffer,
        },
      },
      {
        binding: 7,
        resource: {
          buffer: treeBuffer,
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
      entryPoint: 'main',
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
      entryPoint: 'vert_main',
    },
    fragment: {
      module: device.createShaderModule({
        code: renderShader,
      }),
      entryPoint: 'frag_main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },

    primitive: {
      topology: 'triangle-list',
    },
  });

  let rendered = false;
  let started = false;

  const tileBufferArray = new ArrayBuffer(12);
  let elem = new Int32Array(tileBufferArray, 0, 3);
  elem[0] = 0;
  elem[1] = 0;
  elem[2] = 1;
  let tilesr = 0;

  const frame = async () => {
    device.queue.writeBuffer(tileBuffer, 0, tileBufferArray);

    const compute = async () => {
      if (!rendered) {
        const commandEncoder = device.createCommandEncoder();
        const rayTracePass = commandEncoder.beginComputePass();
        rayTracePass.setPipeline(rayTracingPipeline);
        rayTracePass.setBindGroup(0, rayTracingBindGroup);
        rayTracePass.dispatchWorkgroups(Math.ceil(WIDTH), Math.ceil(HEIGHT), 1);
        rayTracePass.end();
        tilesr = tilesr + 1;
        elem[0] = elem[0] + 1200;
        //console.log(elem[1])
        //console.log("Sned tile number: ", tilesr)
        if (elem[0] == 1200) {
          elem[0] = 0;
          elem[1] = elem[1] + 800;
          if (elem[1] == 800) {
            elem[1] = 0;
            elem[2] = elem[2] + 1;
            //rendered = true;
          }
        }
        const commands = commandEncoder.finish();
        device.queue.submit([commands]);
      }
    };

    if (!started) {
      compute();
      started = false;
    }

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
    passEncoder.setPipeline(screenPipeline);
    passEncoder.setBindGroup(0, screenBindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    device.queue.onSubmittedWorkDone();
    //compute();
    requestAnimationFrame(frame);
  };

  return {
    device,
    cameraBuffer,
    viewMatBuffer,
    worldBuffer,
    createdBuffer,
    frame,
    tileBuffer,
    treeBuffer,
  };
};

const functions = { init };

export default functions;
