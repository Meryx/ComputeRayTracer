const getUsageFromList = (usageList) => {
  return usageList
    .map((e) => window.GPUTextureUsage[e])
    .reduce((accumulator, e) => accumulator | e);
};
const createTextureAndReturnView = ({ device, size, format, usageList }) => {
  const usage = getUsageFromList(usageList);
  const texture = device.createTexture({
    size,
    format,
    usage,
  });
  return texture.createView();
};

export default createTextureAndReturnView;
