import Program from './Program';
import createSampler from './Sampler';
import createShader from './Shader';
import WGSLScreenShader from '../shaders/screen_shader.wgsl';

import { PRESENTATION_FORMAT } from './Constants';

const createQuadProgram = ({ device, framebuffer }) => {
  const quadProgram = new Program();
  quadProgram.setDevice(device);

  const framebufferSamplerDescriptor = {
    device,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'nearest',
    mipmapFilter: 'nearest',
    maxAnisotropy: 1,
  };
  const framebufferSampler = createSampler(framebufferSamplerDescriptor);

  const vertexShaderDescriptor = {
    device,
    code: WGSLScreenShader,
    entryPoint: 'vert_main',
  };
  const fragmentShaderDescriptor = {
    device,
    code: WGSLScreenShader,
    entryPoint: 'frag_main',
    targets: [
      {
        format: PRESENTATION_FORMAT,
      },
    ],
  };
  const vertexShader = createShader(vertexShaderDescriptor);
  const fragmentShader = createShader(fragmentShaderDescriptor);

  quadProgram.addEntry(framebufferSampler, 'FRAGMENT', 'sampler');
  quadProgram.addEntry(framebuffer, 'FRAGMENT', 'texture');
  quadProgram.addShader(vertexShader, 'vertex');
  quadProgram.addShader(fragmentShader, 'fragment');
  quadProgram.compileRenderProgram();

  return quadProgram;
};

export default createQuadProgram;
