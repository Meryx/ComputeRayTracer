const createShader = ({ device, code, entryPoint, targets }) => {
  const module = device.createShaderModule({
    code,
  });
  const shader = {
    module,
    entryPoint,
    targets,
  };
  return shader;
};

export default createShader;
