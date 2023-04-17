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
let textureBuffer;

const loadTextureIntoRayTraceProgram = ({ data, width, height }) => {
  raytracer.device.queue.writeBuffer(textureBuffer, 0, data.buffer);
  console.log(data.buffer);
};

const loadMeshIntoRayTraceProgram = ({
  triangleBuffer,
  vertexBuffer,
  numOfTriangles,
}) => {
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
    usageList: ['COPY_DST', 'UNIFORM'],
  };
  isMeshLoadedBuffer = createBuffer(isMeshLoadedBufferDescriptor);

  raytracer.device.queue.writeBuffer(
    isMeshLoadedBuffer,
    0,
    isMeshLoadedBufferArray
  );

  raytracer.addEntry(isMeshLoadedBuffer, 'COMPUTE', 'buffer', {
    type: 'uniform',
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

  const perm_xBufferDescriptor = {
    device,
    size: 256 * 4,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const perm_xBuffer = createBuffer(perm_xBufferDescriptor);
  raytracer.addEntry(perm_xBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  const perm_yBufferDescriptor = {
    device,
    size: 256 * 4,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const perm_yBuffer = createBuffer(perm_yBufferDescriptor);
  raytracer.addEntry(perm_yBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  const perm_zBufferDescriptor = {
    device,
    size: 256 * 4,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const perm_zBuffer = createBuffer(perm_zBufferDescriptor);
  raytracer.addEntry(perm_zBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  const ranfloatBufferDescriptor = {
    device,
    size: 256 * 16,
    usageList: ['COPY_DST', 'STORAGE'],
  };
  const ranfloatBuffer = createBuffer(ranfloatBufferDescriptor);
  raytracer.addEntry(ranfloatBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  const textureBufferDescriptor = {
    device,
    size: 2097152,
    usageList: ['COPY_DST', 'STORAGE'],
  };

  textureBuffer = createBuffer(textureBufferDescriptor);
  raytracer.addEntry(textureBuffer, 'COMPUTE', 'buffer', {
    type: 'read-only-storage',
  });

  const perm_xBufferArray = new ArrayBuffer(256 * 4);
  const perm_yBufferArray = new ArrayBuffer(256 * 4);
  const perm_zBufferArray = new ArrayBuffer(256 * 4);
  const ranfloatBufferArray = new ArrayBuffer(256 * 16);

  const perm_xArray = new Int32Array(perm_xBufferArray);
  const perm_yArray = new Int32Array(perm_yBufferArray);
  const perm_zArray = new Int32Array(perm_zBufferArray);
  const ranfloatArray = new Float32Array(ranfloatBufferArray);

  const ranf = (min, max) => {
    return min + Math.random() * (max - min);
  };
  for (let x = 0; x < 256; x++) {
    ranfloatArray[x * 4] = ranf(-1, 1);
    ranfloatArray[x * 4 + 1] = ranf(-1, 1);
    ranfloatArray[x * 4 + 2] = ranf(-1, 1);
  }

  perlin_generate_perm(perm_xArray);
  perlin_generate_perm(perm_yArray);
  perlin_generate_perm(perm_zArray);

  raytracer.device.queue.writeBuffer(perm_xBuffer, 0, perm_xBufferArray);

  raytracer.device.queue.writeBuffer(perm_yBuffer, 0, perm_yBufferArray);

  raytracer.device.queue.writeBuffer(perm_zBuffer, 0, perm_zBufferArray);

  raytracer.device.queue.writeBuffer(ranfloatBuffer, 0, ranfloatBufferArray);

  raytracer.addShader(computeShader, 'compute');
  raytracer.compileComputeProgram();
  return raytracer;
};

const perlin_generate_perm = (arr) => {
  for (let x = 0; x < 256; x++) {
    arr[x] = x;
  }
  for (let x = 255; x > 0; x--) {
    let t = randIntRange(0, x);
    let temp = arr[x];
    arr[x] = arr[t];
    arr[t] = temp;
  }
};

const randIntRange = (min, max) => {
  return Math.floor(min + Math.random() * (max - min));
};

export {
  createRaytraceProgram,
  loadMeshIntoRayTraceProgram,
  loadTextureIntoRayTraceProgram,
};
