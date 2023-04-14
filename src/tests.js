let processed = 0;
function rayBVHIntersection() {
  let hitRecord = {};
  let temp1 = {};
  let temp2 = {};
  let rec = { hit: false };
  let stack = new Array(20);
  let top = -1;
  top += 1;
  stack[top] = 0;
  //let t_mx = tMax;

  while (top >= 0) {
    let currentNodeIndex = stack[top];
    top -= 1;

    let currentNode = nodes[currentNodeIndex];
    //hitRecord = rayBoundingBoxIntersection(ray, currentNode, tMin, t_mx);

    if (true) {
      if (currentNode.isLeaf === 1) {
        console.log('Leaf node: ', currentNodeIndex);
      } else {
        console.log('non-leaf node: ', currentNodeIndex);
        if (currentNode.left !== -1) {
          top += 1;
          stack[top] = currentNode.left;
        }
        if (currentNode.right !== -1) {
          top += 1;
          stack[top] = currentNode.right;
        }
      }
    } else {
      console.log('Node not hot', currentNodeIndex);
    }
  }

  if (rec.hit) {
    return rec;
  }

  hitRecord.hit = false;
  return hitRecord;
}

const nodes = [
  {
    min: [-1000, -2001, -1000],
    max: [1000, 2, 1000],
    isLeaf: 0,
    left: 2,
    right: 1,
  },
  {
    min: [-10.871396315464704, 0, -10.003823760993159],
    max: [-9.980549215180341, 0.4, -5.299445735697941],
    isLeaf: 0,
    left: 4,
    right: 3,
  },
  {
    min: [-1000, -2001, -1000],
    max: [1000, 2, 1000],
    isLeaf: 0,
    left: 6,
    right: 5,
  },
  {
    min: [-10.535143835167318, 0, -8.399206119591245],
    max: [-9.980549215180341, 0.4, -6.3357745206933185],
    isLeaf: 0,
    left: 8,
    right: 7,
  },
  {
    min: [-10.871396315464704, 0, -10.003823760993159],
    max: [-10.272358847906363, 0.4, -5.299445735697941],
    left: 5,
    right: 9,
    isLeaf: 1,
  },
  {
    min: [-5, 0, -1],
    max: [5, 2, 1],
    isLeaf: 0,
    left: 10,
    right: 9,
  },
  {
    min: [-1000, -2001, -1000],
    max: [1000, 0.4, 1000],
    left: 4,
    right: 1,
    isLeaf: 1,
  },
  {
    min: [-10.460489163811674, 0, -8.399206119591245],
    max: [-9.980549215180341, 0.4, -6.3357745206933185],
    left: 6,
    right: 8,
    isLeaf: 1,
  },
  {
    min: [-10.535143835167318, 0, -7.541532840479699],
    max: [-10.13514383516732, 0.4, -7.141532840479699],
    left: 7,
    right: 7,
    isLeaf: 1,
  },
  {
    min: [-1, 0, -1],
    max: [5, 2, 1],
    left: 0,
    right: 3,
    isLeaf: 1,
  },
  {
    min: [-5, 0, -1],
    max: [-3, 2, 1],
    left: 2,
    right: 2,
    isLeaf: 1,
  },
];

rayBVHIntersection();
