import { vec3 } from 'gl-matrix';

function getVectorLength(vec1, vec2) {
  const dx = vec2[0] - vec1[0];
  const dy = vec2[1] - vec1[1];
  const dz = vec2[2] - vec1[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

const subtractVecs = (vec1, vec2) => {
  return [vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]];
};

const addVecs = (vec1, vec2) => {
  return [vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2]];
};

const computeSphereBoundingBox = (shape) => {
  let min = subtractVecs(shape.point, [
    shape.dimension,
    shape.dimension,
    shape.dimension,
  ]);
  let max = addVecs(shape.point, [
    shape.dimension,
    shape.dimension,
    shape.dimension,
  ]);
  shape.min = min;
  shape.max = max;
  return shape;
};

const init = () => {
  const bufferSize = 500 * 96;
  let shapes = [];

  let element = 1;

  const arrayBuffer = new ArrayBuffer(bufferSize);
  // let shapeArray = new Uint32Array(arrayBuffer, element * 0, 1); // Offset: 0
  // let pointArray = new Float32Array(arrayBuffer,element * 16, 3); // Offset: 16
  // let albedoArray = new Float32Array(arrayBuffer, element *32, 3); // Offset: 32
  // let dimensionArray = new Float32Array(arrayBuffer, element * 44, 1); // Offset: 48
  // let materialArray = new Uint32Array(arrayBuffer, element * 48, 1); // Offset: 52
  // let fuzzyArray = new Float32Array(arrayBuffer, element * 52, 1); // Offset: 56
  // let etatArray = new Float32Array(arrayBuffer, element * 56, 1); // Offset: 60

  // shapeArray[0] = shape;
  // pointArray.set(point);
  // albedoArray.set(albedo);
  // dimensionArray[0] = dimension;
  // materialArray[0] = material;
  // fuzzyArray[0] = fuziness;
  // etatArray[0] = etat;

  element = 0;
  let created = 0;

  let groundSphere = {};
  groundSphere.shape = 1;
  groundSphere.point = [0, -1001, 0];
  groundSphere.dimension = 1000;
  groundSphere.albedo = [0.8, 0.8, 0.8];
  groundSphere.material = 1;
  groundSphere.fuziness = 0;
  groundSphere.etat = 1.0;

  let metalSphere = {};
  metalSphere.shape = 1;
  metalSphere.point = [4, 1, 0];
  metalSphere.dimension = 1.0;
  metalSphere.albedo = [0.7, 0.6, 0.5];
  metalSphere.material = 2;
  metalSphere.fuziness = 0;
  metalSphere.etat = 1.0;

  let glassSphere = {};
  glassSphere.shape = 1;
  glassSphere.point = [0, 1, 0];
  glassSphere.dimension = 1.0;
  glassSphere.albedo = [1, 1, 1];
  glassSphere.material = 3;
  glassSphere.fuziness = 0;
  glassSphere.etat = 1.8;

  let diffuseSphere = {};
  diffuseSphere.shape = 1;
  diffuseSphere.point = [-4, 1, 0];
  diffuseSphere.dimension = 1.0;
  diffuseSphere.albedo = [0.4, 0.2, 0.1];
  diffuseSphere.material = 1;
  diffuseSphere.fuziness = 0;
  diffuseSphere.etat = 1.0;

  let mainObjs = [metalSphere, groundSphere, diffuseSphere, glassSphere];

  let stride = 96;
  let id = 0;

  for (let i = 0; i < 4; i++) {
    let shapeArray = new Uint32Array(arrayBuffer, element * stride + 0, 1); // Offset: 0
    let pointArray = new Float32Array(arrayBuffer, element * stride + 16, 3); // Offset: 16
    let albedoArray = new Float32Array(arrayBuffer, element * stride + 32, 3); // Offset: 32
    let dimensionArray = new Float32Array(
      arrayBuffer,
      element * stride + 76,
      1
    );
    let materialArray = new Uint32Array(arrayBuffer, element * stride + 80, 1); // Offset: 52
    let fuzzyArray = new Float32Array(arrayBuffer, element * stride + 84, 1); // Offset: 56
    let etatArray = new Float32Array(arrayBuffer, element * stride + 88, 1); // Offset: 60
    mainObjs[i] = computeSphereBoundingBox(mainObjs[i]);
    let { shape, point, albedo, dimension, material, fuziness, etat } =
      mainObjs[i];

    shapeArray[0] = shape;
    pointArray.set(point);
    albedoArray.set(albedo);
    dimensionArray[0] = dimension;
    materialArray[0] = material;
    fuzzyArray[0] = fuziness;
    etatArray[0] = etat;
    mainObjs[i].id = id;
    id = id + 1;

    element = element + 1;
    created = created + 1;
  }

  shapes = [...mainObjs];

  let shape = 1;
  let point = [0, 0, -2];
  let albedo = [0.8, 0.0, 0.0];
  let dimension = 0.5;
  let material = 1;
  let fuziness = 0;
  let etat = 1.0;

  for (let a = -11; a < 11; a++) {
    for (let b = -11; b < 11; b++) {
      const choose_mat = Math.random();
      const center = [a + 0.9 * Math.random(), 0.2, b + 0.9 * Math.random()];

      let length = getVectorLength(center, [4, 0.2, 0]);
      if (length > 0.9) {
        let sphere_material;

        if (choose_mat < 0.8) {
          // diffuse

          let randomDrop = Math.random();

          shape = 1;
          point = center;
          albedo = [
            Math.random() * Math.random(),
            Math.random() * Math.random(),
            Math.random() * Math.random(),
          ];
          dimension = 0.2;
          material = 1;
          fuziness = 0;
          etat = 1.0;
        } else if (choose_mat < 0.95) {
          // metal

          let randomDrop = Math.random();

          shape = 1;
          point = center;
          albedo = [
            Math.random() / 2 + 0.5,
            Math.random() / 2 + 0.5,
            Math.random() / 2 + 0.5,
          ];
          dimension = 0.2;
          material = 2;
          fuziness = Math.random() * 0.5;
          etat = 1.0;
        } else {
          // glass
          shape = 1;
          point = center;
          albedo = [1, 1, 1];
          dimension = 0.2;
          material = 3;
          fuziness = 0;
          etat = 1.5;
        }

        shapes = [
          ...shapes.slice(),
          { shape, point, albedo, dimension, material, fuziness, etat },
        ];
        shapes[shapes.length - 1] = computeSphereBoundingBox(
          shapes[shapes.length - 1]
        );
        shapes[shapes.length - 1].id = id;
        id = id + 1;

        let shapeArray = new Uint32Array(arrayBuffer, element * stride + 0, 1); // Offset: 0
        let pointArray = new Float32Array(
          arrayBuffer,
          element * stride + 16,
          3
        ); // Offset: 16
        let albedoArray = new Float32Array(
          arrayBuffer,
          element * stride + 32,
          3
        ); // Offset: 32
        let dimensionArray = new Float32Array(
          arrayBuffer,
          element * stride + 76,
          1
        ); // Offset: 48
        let materialArray = new Uint32Array(
          arrayBuffer,
          element * stride + 80,
          1
        ); // Offset: 52
        let fuzzyArray = new Float32Array(
          arrayBuffer,
          element * stride + 84,
          1
        ); // Offset: 56
        let etatArray = new Float32Array(arrayBuffer, element * stride + 88, 1); // Offset: 60

        shapeArray[0] = shape;
        pointArray.set(point);
        albedoArray.set(albedo);
        dimensionArray[0] = dimension;
        materialArray[0] = material;
        fuzzyArray[0] = fuziness;
        etatArray[0] = etat;

        element = element + 1;
        created = created + 1;
      }
    }
  }

  const createdBufferArray = new ArrayBuffer(4);
  let elem = new Uint32Array(createdBufferArray, 0, 1);
  elem[0] = created;

  let shapescp = shapes.slice();

  const head = createBVH(shapescp, 0, shapescp.length);
  const nodes = constructBVHarray(head);
  console.log(head);
  console.log(nodes);
  console.log(shapes.slice(0, 10));

  elem[0] = nodes.length;

  const bvharraybuffer = new ArrayBuffer(48 * 550);
  stride = 48;
  for (let i = 0; i < nodes.length; i++) {
    let minArray = new Float32Array(bvharraybuffer, stride * i + 0, 3);
    let maxArray = new Float32Array(bvharraybuffer, stride * i + 16, 3);
    let rightArray = new Int32Array(bvharraybuffer, stride * i + 28, 1);
    let leftArray = new Int32Array(bvharraybuffer, stride * i + 32, 1);
    let isLeaf = new Int32Array(bvharraybuffer, stride * i + 36, 1);
    minArray.set(nodes[i].min);
    maxArray.set(nodes[i].max);
    rightArray[0] = nodes[i].right;
    leftArray[0] = nodes[i].left;
    isLeaf[0] = nodes[i].isLeaf;
  }

  for (let i = nodes.length; i < 550; i++) {
    let minArray = new Float32Array(bvharraybuffer, stride * i + 0, 3);
    let maxArray = new Float32Array(bvharraybuffer, stride * i + 16, 3);
    let rightArray = new Int32Array(bvharraybuffer, stride * i + 28, 1);
    let leftArray = new Int32Array(bvharraybuffer, stride * i + 32, 1);
    let isLeaf = new Int32Array(bvharraybuffer, stride * i + 36, 1);
    minArray.set([-1, -1, -1]);
    maxArray.set([-1, -1, -1]);
    rightArray[0] = -1;
    leftArray[0] = -1;
    isLeaf[0] = 0;
  }

  return { arrayBuffer, createdBufferArray, bvharraybuffer };
};

let nonleafcount = 0;

function constructBVHarray(root) {
  let result = [];
  let queue = [root];

  while (queue.length > 0) {
    let currentNode = queue.shift();
    const { min, max, isLeaf } = currentNode;

    if (!currentNode.isLeaf) {
      let right = nonleafcount * 2 + 1;
      let left = nonleafcount * 2 + 2;
      result.push({ min, max, isLeaf: 0, left, right });
      queue.push(currentNode.right);
      queue.push(currentNode.left);
      nonleafcount = nonleafcount + 1;
    } else {
      result.push({ ...currentNode, isLeaf: 1 });
    }
  }

  return result;
}

function sortNodes(shapes, start, end, compare, axis) {
  const s = shapes.slice(start, end).sort((a, b) => {
    return compare(a, b, axis) ? -1 : 1;
  });

  // Apply the interval [start, end) by updating the shapes array in-place.
  shapes.splice(start, end - start, ...s);
}

const compare = (s1, s2, axis) => {
  return s1.min[axis] < s2.min[axis];
};

const mergeBox = (b1, b2) => {
  const [x1, y1, z1] = [b1.min[0], b1.min[1], b1.min[2]];
  const [x2, y2, z2] = [b2.min[0], b2.min[1], b2.min[2]];

  let minx = Math.min(x1, x2);
  let miny = Math.min(y1, y2);
  let minz = Math.min(z1, z2);

  const [X1, Y1, Z1] = [b1.max[0], b1.max[1], b1.max[2]];
  const [X2, Y2, Z2] = [b2.max[0], b2.max[1], b2.max[2]];

  let maxx = Math.max(X1, X2);
  let maxy = Math.max(Y1, Y2);
  let maxz = Math.max(Z1, Z2);

  let isLeaf = false;
  let left = b1,
    right = b2;
  if (Number.isInteger(b1.id)) {
    left = b1.id;
    right = b2.id;
    isLeaf = true;
  }

  const box = {
    min: [minx, miny, minz],
    max: [maxx, maxy, maxz],
    left,
    right,
    isLeaf,
  };

  return box;
};

const createBVH = (shapes, start, end) => {
  const span = end - start;
  let left;
  let right;
  let axis = randomInt(0, 2);
  if (span === 1) {
    left = shapes[start];
    right = shapes[start];
  }
  if (span === 2) {
    if (compare(shapes[start], shapes[start + 1])) {
      left = shapes[start];
      right = shapes[start + 1];
    } else {
      left = shapes[start + 1];
      right = shapes[start];
    }
  }
  if (span > 2) {
    sortNodes(shapes, start, end, compare, axis);
    const mid = start + Math.floor(span / 2);
    left = createBVH(shapes, start, mid);
    right = createBVH(shapes, mid, end);
  }
  return mergeBox(left, right);
};

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

const functions = { init };

export default functions;
