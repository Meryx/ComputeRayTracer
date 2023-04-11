import Program from './Program';
import createShader from './Shader';
import createBuffer from './Buffer';
import WGSLRaytracer from '../shaders/raytracer.wgsl';

import { PRESENTATION_FORMAT } from './Constants';

const raytracer = new Program();

let isMeshLoadedBufferArray;
let isMeshLoadedArray;
let isMeshLoadedBufferDescriptor;
let isMeshLoadedBuffer;

const loadMeshIntoRayTraceProgram = ({
  triangleBuffer,
  vertexBuffer,
  numOfTriangles,
}) => {
  console.log(raytracer.entries);
  raytracer.entries = raytracer.entries.slice(0, 2);

  const numOfTrianglesBufferArray = new ArrayBuffer(4);
  const numOfTrianglesArray = new Uint32Array(numOfTrianglesBufferArray);
  numOfTrianglesArray[0] = numOfTriangles;
  const numOfTrianglesBufferDescriptor = {
    device: raytracer.device,
    size: 4,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const numOfTrianglesBuffer = createBuffer(numOfTrianglesBufferDescriptor);

  raytracer.addEntry(numOfTrianglesBuffer, 'COMPUTE', 'buffer', {
    type: 'storage',
  });

  raytracer.addEntry(triangleBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  raytracer.addEntry(vertexBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  raytracer.device.queue.writeBuffer(
    numOfTrianglesBuffer,
    0,
    numOfTrianglesBufferArray
  );

  raytracer.device.queue.writeBuffer(
    isMeshLoadedBuffer,
    0,
    isMeshLoadedBufferArray
  );

  isMeshLoadedArray = new Uint32Array(isMeshLoadedBufferArray);
  isMeshLoadedArray[0] = 1;
  raytracer.compileComputeProgram();
  raytracer.device.queue.writeBuffer(
    isMeshLoadedBuffer,
    0,
    isMeshLoadedBufferArray
  );
};

const createRaytraceProgram = ({ device, framebuffer }) => {
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

  isMeshLoadedBufferArray = new ArrayBuffer(4);
  isMeshLoadedArray = new Uint32Array(isMeshLoadedBufferArray);
  isMeshLoadedArray[0] = 0;
  isMeshLoadedBufferDescriptor = {
    device,
    size: 4,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  isMeshLoadedBuffer = createBuffer(isMeshLoadedBufferDescriptor);

  raytracer.device.queue.writeBuffer(
    isMeshLoadedBuffer,
    0,
    isMeshLoadedBufferArray
  );

  raytracer.addEntry(isMeshLoadedBuffer, 'COMPUTE', 'buffer', {
    type: 'storage',
  });

  const numOfTrianglesBufferArray = new ArrayBuffer(4);
  const numOfTrianglesArray = new Uint32Array(numOfTrianglesBufferArray);
  numOfTrianglesArray[0] = 0;
  const numOfTrianglesBufferDescriptor = {
    device: raytracer.device,
    size: 4,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const numOfTrianglesBuffer = createBuffer(numOfTrianglesBufferDescriptor);

  raytracer.addEntry(numOfTrianglesBuffer, 'COMPUTE', 'buffer', {
    type: 'storage',
  });

  const triangleBufferDescriptor = {
    device,
    size: 16,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const triangleBuffer = createBuffer(triangleBufferDescriptor);
  raytracer.addEntry(triangleBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  const vertexBufferDescriptor = {
    device,
    size: 16,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const vertexBuffer = createBuffer(vertexBufferDescriptor);
  raytracer.addEntry(vertexBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  raytracer.addShader(computeShader, 'compute');
  raytracer.compileComputeProgram();
  return raytracer;
};

export { createRaytraceProgram, loadMeshIntoRayTraceProgram };
