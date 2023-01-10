import { vec3, mat4 } from "gl-matrix";
const init = (d, ar) => {
  let distance = vec3.create();
  let target = vec3.create();
  let upvector = vec3.create();
  let viewMat = mat4.create();
  let cameraMat = mat4.create();
  let position = vec3.create();
  let orbitX = 0;
  let orbitY = 0;
  let maxOrbitX = 0;
  let minOrbitX = -Math.PI * 0.5;

  vec3.set(distance, 0, 0, d);
  vec3.set(target, 0, 0, 0);
  vec3.set(upvector, 0, 1, 0);

  var temp = mat4.create();
  mat4.translate(temp, temp, distance);

  const viewMatSize = 16 * Float32Array.BYTES_PER_ELEMENT;
  const viewMatBufferArray = new ArrayBuffer(viewMatSize);
  const viewMatBufferView = new Float32Array(viewMatBufferArray);

  mat4.copy(viewMatBufferView, temp);

  let origin = [0.0, 0.0, 1.0];
  let lookAt = [0.0, 0.0, -1.0];
  let up = [0.0, 1.0, 0.0];
  let focalDistance = d;
  let aspectRatio = ar;

  const cameraBufferSize = 11 * Float32Array.BYTES_PER_ELEMENT;
  const cameraBufferArray = new ArrayBuffer(cameraBufferSize);
  const cameraArrayOrigin = new Float32Array(cameraBufferArray, 0, 3);
  const cameraArrayLookat = new Float32Array(cameraBufferArray, 12, 3);
  const cameraArrayUp = new Float32Array(cameraBufferArray, 24, 3);
  const cameraArrayFocalDistance = new Float32Array(cameraBufferArray, 36, 1);
  const cameraArrayAspectRatio = new Float32Array(cameraBufferArray, 40);

  cameraArrayOrigin.set(origin);
  cameraArrayLookat.set(lookAt);
  cameraArrayUp.set(up);
  cameraArrayFocalDistance[0] = focalDistance;
  cameraArrayAspectRatio[0] = aspectRatio;

  const { rotate, dolly } = (() => {
    let deltaX = 0;
    let deltaY = 0;
    let delta = d;
    const rotate = (axis, flag) => {
      if (axis === 1) {
        deltaX = deltaX + flag * 0.1;
        if (deltaX < minOrbitX) {
          deltaX = minOrbitX;
        }
        if (deltaX > maxOrbitX) {
          deltaX = maxOrbitX;
        }
        mat4.identity(temp);
        mat4.rotateY(temp, temp, deltaY);
        mat4.rotateX(temp, temp, deltaX);
        mat4.translate(temp, temp, [...distance.slice(0, 2), delta]);
        mat4.copy(viewMatBufferView, temp);
      } else {
        deltaY = deltaY + flag * 0.1;
        mat4.identity(temp);
        mat4.rotateY(temp, temp, deltaY);
        mat4.rotateX(temp, temp, deltaX);
        mat4.translate(temp, temp, [...distance.slice(0, 2), delta]);
        mat4.copy(viewMatBufferView, temp);
      }
    };
    const dolly = (flag) => {
      delta = delta + (flag / Math.abs(flag)) * 1;
      mat4.identity(temp);
      mat4.rotateY(temp, temp, deltaY);
      mat4.rotateX(temp, temp, deltaX);
      mat4.translate(temp, temp, [...distance.slice(0, 2), delta]);
      mat4.copy(viewMatBufferView, temp);
    };

    return { rotate, dolly };
  })();

  return { cameraBufferArray, viewMatBufferArray, rotate, dolly };
};

const functions = { init };

export default functions;
