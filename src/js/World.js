const init = () => {
  const shape = 1;
  const point = [0, 0, 0];
  const albedo = [0.8, 0, 0];
  const dimension = 0.5;
  const material = 1;
  const fuziness = 0;
  const etat = 1.0;

  const bufferSize = 11 * Float32Array.BYTES_PER_ELEMENT;
  const arrayBuffer = new ArrayBuffer(bufferSize);
  const shapeArray = new Float32Array(arrayBuffer, 0, 1);
  const pointArray = new Float32Array(arrayBuffer, 4, 3);
  const albedoArray = new Float32Array(arrayBuffer, 16, 3);
  const dimensionArray = new Float32Array(arrayBuffer, 28, 1);
  const materialArray = new Float32Array(arrayBuffer, 32, 1);
  const fuzzyArray = new Float32Array(arrayBuffer, 36, 1);
  const etatArray = new Float32Array(arrayBuffer, 40, 1);

  shapeArray[0] = shape;
  pointArray.set(point);
  albedoArray.set(albedo);
  dimensionArray[0] = dimension;
  materialArray[0] = material;
  fuzzyArray[0] = fuziness;
  etatArray[0] = etat;

  return { arrayBuffer };
};

const functions = { init };

export default functions;
