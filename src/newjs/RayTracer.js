import Program from './Program';
import createShader from './Shader';
import WGSLRaytracer from '../shaders/raytracer.wgsl';

import { PRESENTATION_FORMAT } from './Constants';

const createRaytraceProgram = ({ device, framebuffer }) => {
  const raytracer = new Program();
  raytracer.setDevice(device);

  const computeShaderDescriptor = {
    device,
    code: WGSLRaytracer,
    entryPoint: 'main',
  };
  const computeShader = createShader(computeShaderDescriptor);

  raytracer.addEntry(framebuffer, 'COMPUTE', 'storageTexture', {
    access: 'write-only',
    format: PRESENTATION_FORMAT,
    viewDimension: '2d',
  });
  raytracer.addShader(computeShader, 'compute');
  raytracer.compileComputeProgram('');
  return raytracer;
};

export default createRaytraceProgram;
