import createRenderPipelineAndBindGroup from './RenderPipeline';
import createComputePipelineAndBindGroup from './ComputePipeline';

export default class Program {
  constructor() {
    this.entries = [];
    this.shaders = {};
    this.device = null;
    this.renderPipeline = null;
    this.renderBindGroup = null;
  }

  getPipeline() {
    return this.renderPipeline;
  }

  getBindGroup() {
    return this.renderBindGroup;
  }

  setDevice(device) {
    this.device = device;
  }

  addEntry(entry, visibility, resourceType, resourceDescriptor) {
    const e = {
      visibility,
      resourceType,
      resource: entry,
      resourceDescriptor,
    };
    this.entries.push(e);
  }

  addShader(shader, type) {
    this.shaders[type] = shader;
  }

  compileRenderProgram(topology) {
    const rendererPipelineDescriptor = {
      device: this.device,
      entries: this.entries,
      ...this.shaders,
      topology: topology,
    };
    const { pipeline, bindGroup } = createRenderPipelineAndBindGroup(
      rendererPipelineDescriptor
    );
    this.renderBindGroup = bindGroup;
    this.renderPipeline = pipeline;
  }

  compileComputeProgram() {
    const rendererPipelineDescriptor = {
      device: this.device,
      entries: this.entries,
      ...this.shaders,
    };
    const { pipeline, bindGroup } = createComputePipelineAndBindGroup(
      rendererPipelineDescriptor
    );
    this.renderBindGroup = bindGroup;
    this.renderPipeline = pipeline;
  }
}
