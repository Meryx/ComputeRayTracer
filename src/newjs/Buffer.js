const getUsageFromList = (usageList) => {
  return usageList
    .map((e) => window.GPUBufferUsage[e])
    .reduce((accumulator, e) => accumulator | e);
};
const createBuffer = ({ device, size, usageList }) => {
  const usage = getUsageFromList(usageList);
  const buffer = device.createBuffer({
    size,
    usage,
  });
  return buffer;
};

export default createBuffer;
