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
      resource:
        e.resourceType === 'buffer'
          ? {
              [e.resourceType]: e.resource,
            }
          : e.resource,
    };
  });
};

const createComputePipelineAndBindGroup = ({ device, entries, compute }) => {
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

  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute,
  });

  return { pipeline, bindGroup };
};
export default createComputePipelineAndBindGroup;
