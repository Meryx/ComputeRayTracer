import { vec3 } from "gl-matrix";

function getVectorLength(vec1, vec2) {
  const dx = vec2[0] - vec1[0];
  const dy = vec2[1] - vec1[1];
  const dz = vec2[2] - vec1[2];
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

const init = () => {

  




  const bufferSize = 484 * 16 * Float32Array.BYTES_PER_ELEMENT;

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
  groundSphere.shape = 2;
  groundSphere.point = [0, 0, 0];
  groundSphere.dimension = 1;
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
  glassSphere.etat = 1.5;

  let diffuseSphere = {};
  diffuseSphere.shape = 1;
  diffuseSphere.point = [-4, 1, 0];
  diffuseSphere.dimension = 1.0;
  diffuseSphere.albedo = [0.4, 0.2, 0.1];
  diffuseSphere.material = 1;
  diffuseSphere.fuziness = 0;
  diffuseSphere.etat = 1.0;

  let mainObjs = [groundSphere, diffuseSphere, glassSphere, metalSphere];

  for(let i = 0; i < 4; i++)
  {
        let shapeArray = new Uint32Array(arrayBuffer, element * 64 + 0, 1); // Offset: 0
        let pointArray = new Float32Array(arrayBuffer,element * 64 + 16, 3); // Offset: 16
        let albedoArray = new Float32Array(arrayBuffer, element * 64 + 32, 3); // Offset: 32
        let dimensionArray = new Float32Array(arrayBuffer, element * 64 + 44, 1); // Offset: 48
        let materialArray = new Uint32Array(arrayBuffer, element * 64 + 48, 1); // Offset: 52
        let fuzzyArray = new Float32Array(arrayBuffer, element * 64 + 52, 1); // Offset: 56
        let etatArray = new Float32Array(arrayBuffer, element * 64 + 56, 1); // Offset: 60

        let {shape, point, albedo, dimension, material, fuziness, etat} = mainObjs[i];

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
      const center = [a + 0.9*Math.random(), 0.2, b + 0.9*Math.random()];

      let length = getVectorLength(center, [4, 0.2, 0])
      if (length > 0.9) {
        let sphere_material;
  
        if (choose_mat < 0.8) {
          // diffuse

          let randomDrop = Math.random();
          if(randomDrop < 0.50){
            continue;
          }

          shape = 1;
          point = center;
          albedo = [Math.random() * Math.random(), Math.random() * Math.random(), Math.random() * Math.random()];
          dimension = 0.2;
          material = 1;
          fuziness = 0;
          etat = 1.0;

        } else if (choose_mat < 0.95) {
          // metal

          let randomDrop = Math.random();
          if(randomDrop < 0.50){
            continue;
          }

          shape = 1;
          point = center;
          albedo = [Math.random() / 2 + 0.5, Math.random() / 2 + 0.5, Math.random() / 2 + 0.5];
          dimension = 0.2;
          material = 2;
          fuziness = Math.random() * 0.5;
          etat = 1.0;

        } else {
          // glass
          shape = 1;
          point = center;
          albedo = [1,1,1];
          dimension = 0.2;
          material = 3;
          fuziness = 0;
          etat = 1.5;

        }

        
        let shapeArray = new Uint32Array(arrayBuffer, element * 64 + 0, 1); // Offset: 0
        let pointArray = new Float32Array(arrayBuffer,element * 64 + 16, 3); // Offset: 16
        let albedoArray = new Float32Array(arrayBuffer, element * 64 + 32, 3); // Offset: 32
        let dimensionArray = new Float32Array(arrayBuffer, element * 64 + 44, 1); // Offset: 48
        let materialArray = new Uint32Array(arrayBuffer, element * 64 + 48, 1); // Offset: 52
        let fuzzyArray = new Float32Array(arrayBuffer, element * 64 + 52, 1); // Offset: 56
        let etatArray = new Float32Array(arrayBuffer, element * 64 + 56, 1); // Offset: 60

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
  console.log(created)

  return { arrayBuffer, createdBufferArray};
};

const functions = { init };

export default functions;
