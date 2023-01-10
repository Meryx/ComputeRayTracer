import React, { useEffect } from "react";
import GPU from "./js/GPU";
import Camera from "./js/Camera";
import World from "./js/World";
import { mat4 } from "gl-matrix";
import "./App.css";

const ASPECT_RATIO = 3 / 2;
const WIDTH = 1200;
const HEIGHT = Math.floor(WIDTH / ASPECT_RATIO);
const DISTANCE = 1;

function App() {
  useEffect(() => {
    const setup = async () => {
      const { device, cameraBuffer, viewMatBuffer, worldBuffer } =
        await GPU.init(WIDTH, HEIGHT);
      const { cameraBufferArray, viewMatBufferArray, rotate, dolly } =
        Camera.init(DISTANCE, ASPECT_RATIO);

      const { arrayBuffer } = World.init();

      document.addEventListener("keydown", (event) => {
        if (event.key === "d") {
          rotate(2, 1);
          device.queue.writeBuffer(viewMatBuffer, 0, viewMatBufferArray);
        }
        if (event.key === "w") {
          rotate(1, 1);
          device.queue.writeBuffer(viewMatBuffer, 0, viewMatBufferArray);
        }
        if (event.key === "a") {
          rotate(2, -1);
          device.queue.writeBuffer(viewMatBuffer, 0, viewMatBufferArray);
        }
        if (event.key === "s") {
          rotate(1, -1);
          device.queue.writeBuffer(viewMatBuffer, 0, viewMatBufferArray);
        }
      });

      document.addEventListener("mousewheel", (event) => {
        dolly(event.deltaY);
        device.queue.writeBuffer(viewMatBuffer, 0, viewMatBufferArray);
      });

      device.queue.writeBuffer(cameraBuffer, 0, cameraBufferArray);
      device.queue.writeBuffer(viewMatBuffer, 0, viewMatBufferArray);
      device.queue.writeBuffer(worldBuffer, 0, arrayBuffer);
    };

    setup();
  });

  return (
    <div className="App">
      <header className="App-header">
        <canvas id="canvas" width={WIDTH} height={HEIGHT}></canvas>
      </header>
    </div>
  );
}

export default App;
