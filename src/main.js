import { render } from 'react-dom';
import TextureRenderShaderCode from './shaders/TextureRenderShader.wgsl';
import ComputeShaderCode from './shaders/ComputeShader.wgsl';

class Renderer {
  constructor({ device, context, integrator, scene }) {
    this.device = device;
    this.context = context;
    this.integrator = integrator;
    this.scene = scene;
  }

  render({ renderPipeline, renderPipelineBindGroup, computePipeline, computePipelineBindGroup }) {
    const { device, context, integrator, scene } = this;
    const { queue } = device;
    const frame = () => {
      const texture = context.getCurrentTexture();
      const textureView = texture.createView();
      const renderPassDescriptor = {
        colorAttachments: [
          {
            view: textureView,
            clearValue: { r: 0.95, g: 0.34, b: 0.5, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      };
      const commandEncoder = device.createCommandEncoder();
      const computePassEncoder = commandEncoder.beginComputePass();
      computePassEncoder.setPipeline(computePipeline);
      computePassEncoder.setBindGroup(0, computePipelineBindGroup);
      computePassEncoder.dispatchWorkgroups(1920, 1080, 1);
      computePassEncoder.end();
      const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      renderPassEncoder.setPipeline(renderPipeline);
      renderPassEncoder.setBindGroup(0, renderPipelineBindGroup);
      renderPassEncoder.draw(6, 1, 0, 0);
      renderPassEncoder.end();
      const commands = commandEncoder.finish();
      queue.submit([commands]);
    };

    requestAnimationFrame(frame);
  }
}

const Main = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  device.addEventListener('uncapturederror', (event) => {
    // Re-surface the error, because adding an event listener may silence console logs.
    console.error('A WebGPU error was not captured:', event.error);
  });

  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');

  //const format = navigator.gpu.getPreferredCanvasFormat();
  const format = 'rgba8unorm';
  context.configure({
    device,
    format,
  });

  const framebufferDescriptor = {
    size: {
      width: 1920,
      height: 1080,
    },
    format,
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  };

  const framebuffer = device.createTexture(framebufferDescriptor);
  const framebufferView = framebuffer.createView();

  const framebufferSamplerDescriptor = {
    magFilter: 'linear',
  };
  const framebufferSampler = device.createSampler(framebufferSamplerDescriptor);
  const textureRenderShaderModule = device.createShaderModule({
    code: TextureRenderShaderCode,
  });

  const vertexStage = {
    module: textureRenderShaderModule,
    entryPoint: 'vert_main',
  };

  const fragmentStage = {
    module: textureRenderShaderModule,
    entryPoint: 'frag_main',
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

  const renderPipelineBingGroupLayout = device.createBindGroupLayout(renderPipelinBindGroupLayoutDescriptor);

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

  const renderPipelineBindGroup = device.createBindGroup(renderPipelinBindGroupDescriptor);

  const renderPipelineLayoutDescriptor = {
    bindGroupLayouts: [renderPipelineBingGroupLayout],
  };

  const renderPipelineLayout = device.createPipelineLayout(renderPipelineLayoutDescriptor);

  const renderPipelineDescriptor = {
    layout: renderPipelineLayout,
    vertex: vertexStage,
    fragment: fragmentStage,
  };

  const renderPipeline = device.createRenderPipeline(renderPipelineDescriptor);

  const computePipelineBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: 'write-only',
          format,
          viewDimension: '2d',
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
      entryPoint: 'main',
    },
  });

  const renderer = new Renderer({ device, context });
  renderer.render({ renderPipeline, renderPipelineBindGroup, computePipeline, computePipelineBindGroup });
};

export default Renderer;
export { Main };
