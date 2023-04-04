import { PRESENTATION_FORMAT, ALPHA_MODE } from './Constants';

const requestDevice = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  return device;
};

const getCanvasElement = () => {
  return document.getElementById('canvas');
};

const getContext = (canvas) => {
  return canvas.getContext('webgpu');
};

const initGPUAndReturnDeviceAndContext = async () => {
  try {
    const device = await requestDevice();
    const canvas = getCanvasElement();
    const context = getContext(canvas);
    context.configure({
      device,
      format: PRESENTATION_FORMAT,
      alphaMode: ALPHA_MODE,
    });
    return { device, context };
  } catch ({ message }) {
    console.log(`Failed to retrieve GPU device with message: ${message}`);
  }
};

export default initGPUAndReturnDeviceAndContext;
