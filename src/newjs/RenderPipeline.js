const createBindGroupLayoutEntries = (entries) => {
  return entries.map((e, i) => {
    return {
      binding: i,
      visibility: window.GPUShaderStage[e.visibility],
      [e.resourceType]: { ...e.resourceDescriptor },
    };
  });
};

const createBindGroupEntries = (entries) => {
  return entries.map((e, i) => {
    return {
      binding: i,
      resource: e.resource,
    };
  });
};

const createRenderPipelineAndBindGroup = ({
  device,
  entries,
  vertex,
  fragment,
  topology,
}) => {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: createBindGroupLayoutEntries(entries),
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: createBindGroupEntries(entries),
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex,
    fragment,
    primitive: {
      topology,
    },
  });

  return { pipeline, bindGroup };
};
export default createRenderPipelineAndBindGroup;
