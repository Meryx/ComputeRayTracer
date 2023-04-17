@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> isMeshLoaded: u32;
@group(0) @binding(2) var<storage, read_write> numOfTriangles: u32;
@group(0) @binding(3) var<storage> triangles: array<vec3<u32>>;
@group(0) @binding(4) var<storage> vertices: array<vec3<f32>>;
@group(0) @binding(5) var<storage> perm_x: array<i32, 256>;
@group(0) @binding(6) var<storage> perm_y: array<i32, 256>;
@group(0) @binding(7) var<storage> perm_z: array<i32, 256>;
@group(0) @binding(8) var<storage> ranfloat: array<vec3<f32>, 256>;
@group(0) @binding(9) var<storage> texture: array<u32, 524288>;


/* Numerical constants */
const PI: f32 = 3.14159265359;
const INFINITY : f32 = 3.40282346638528859812e+38f;
const NUM_OF_OBJECTS : u32 = 2;
const MAX_STACK_SIZE : u32 = 256;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    time: f32,
}


const SPHERE : u32 = 1;
const PLANE : u32 = 2;
const TRIANGLE : u32 = 3;
const BOUNDINGBOX : u32 = 4;

struct Shape {
    shape: u32,
    point: vec3<f32>,
    point1: vec3<f32>,
    albedo: vec3<f32>,
    dimension: f32,
    material: u32,
    fuzziness: f32,
    etat: f32,
    normal: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    v3: vec3<f32>,
    time0: f32,
    time1: f32,
    min: vec3<f32>,
    max: vec3<f32>,
    created: bool,
    isTextured: u32,
}

struct BVHNode {
    box: Shape,
    right: i32,
    left: i32,
    index: i32,
    isLeaf: bool,
    shape: Shape,
}

struct HitRecord { 
    point: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    u: f32,
    v: f32,
    hit: bool,
    frontFace: bool,
    material: u32,
    albedo: vec3<f32>,
    fuzziness: f32,
    etat: f32,
}

struct Camera {
    origin :vec3<f32>,
    horizontal :vec3<f32>,
    vertical :vec3<f32>,
    lowerLeftCorner :vec3<f32>,
    lensRadius : f32,
    w: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
    time0: f32,
    time1: f32,
}



fn setupCamera() 
{

    let focalDistance: f32 = 1.0;
    let aspectRatio: f32 = 3.0 / 2.0;
    let fov : f32 = 20;
    let theta = degreesToRadians(fov);
    let h = tan(theta/2);
    let viewportHeight = 2.0 * h;
    let viewportWidth = aspectRatio * viewportHeight;

    let lookFrom = vec3<f32>(13,2,3);
    let lookAt = vec3<f32>(0,0,0);
    let aperture = 0.001;
    let focusDist : f32 = 2;
    let vup = vec3<f32>(0,1,0);
    let w = normalize(lookFrom - lookAt);
    let u = normalize(cross(vup, w));
    let v = cross(w,u);



    worldCamera.origin = lookFrom;
    worldCamera.horizontal = focusDist * viewportWidth * u;
    worldCamera.vertical = focusDist * viewportHeight * v;
    worldCamera.lowerLeftCorner = worldCamera.origin - worldCamera.horizontal/2 - worldCamera.vertical/2 - focusDist * w;
    worldCamera.lensRadius = aperture / 2;
    worldCamera.w = w;
    worldCamera.u = u;
    worldCamera.v = v;
    worldCamera.time0 = 0;
    worldCamera.time1 = 1;
}

var<private> objects : array<Shape, NUM_OF_OBJECTS>;
var<private> nodes : array<BVHNode, 2 * NUM_OF_OBJECTS>;
var<private> worldCamera : Camera;
var<private> redSphere : Shape;
var<private> grayPlane : Shape;
var<private> bigPlane1 : Shape;
var<private> bigPlane2 : Shape;
var<private> blueBoundingBox : Shape;

fn setupWorld() {
    redSphere.shape = SPHERE;
    redSphere.point = vec3<f32>(0, 0, -1.5);
    redSphere.point1 = vec3<f32>(0.0, 0, -1.5);
    redSphere.time0 = 0;
    redSphere.time1 = 1;
    redSphere.dimension = 0.7;
    
    redSphere.material = 1;
    redSphere.isTextured = 0;

    redSphere.albedo = vec3<f32>(0.0, 0.8, 0.0);
    //objects[0] = redSphere;

    grayPlane.shape = SPHERE;
    grayPlane.point = vec3<f32>(0.0, -1001, -1.5f);
    grayPlane.point1 = vec3<f32>(0.0, -1001, -1.5f);
    grayPlane.albedo = vec3<f32>(0.7, 0.7, 0.7);
    grayPlane.material = 1;
    grayPlane.time0 = 0;
    grayPlane.time1 = 1;
    grayPlane.dimension = 1000.0f;
    //grayPlane.normal = vec3<f32>(0.0f, 1.0f, 0.0f);
    grayPlane.fuzziness = 0.1;
    grayPlane.isTextured = 1;
    //objects[1] = grayPlane;

    bigPlane2.shape = SPHERE;
    bigPlane2.point = vec3<f32>(0.0, 1, -0.5f);
    bigPlane2.point1 = vec3<f32>(0.0, 1, -0.5f);
    bigPlane2.albedo = vec3<f32>(1,0.4,0);
    bigPlane2.material = 1;
    bigPlane2.time0 = 0;
    bigPlane2.time1 = 1;
    bigPlane2.dimension = 2.0f;
    bigPlane2.etat = 1.5;
    //grayPlane.normal = vec3<f32>(0.0f, 1.0f, 0.0f);
    bigPlane2.fuzziness = 0;
    bigPlane2.isTextured = 2;
    objects[0] = bigPlane2;

    bigPlane1.shape = SPHERE;
    bigPlane1.point = vec3<f32>(0.0, -1001, -1.5f);
    bigPlane1.point1 = vec3<f32>(0.0, -1001, -1.5f);
    bigPlane1.albedo = vec3<f32>(0.7, 0.7, 0.7);
    bigPlane1.material = 1;
    bigPlane1.time0 = 0;
    bigPlane1.time1 = 1;
    bigPlane1.dimension = 1000.0f;
    //grayPlane.normal = vec3<f32>(0.0f, 1.0f, 0.0f);
    bigPlane1.fuzziness = 0.1;
    bigPlane1.isTextured = 1;
    objects[1] = bigPlane1;

    blueBoundingBox.min = vec3<f32>(-2.5, -1.1, -2.0);
    blueBoundingBox.max = vec3<f32>(1.5, -0.8, -1.0);
    blueBoundingBox.albedo = vec3<f32>(0.0, 0.0, 1.0);
    blueBoundingBox.shape = BOUNDINGBOX;
    blueBoundingBox.material = 4;
    //objects[2] = blueBoundingBox;
    //objects[0] = blueBoundingBox;

}

fn lengthSquared(v : vec3<f32>) -> f32 {
    return dot(v, v);
}

fn degreesToRadians(angle : f32) -> f32
{
    return angle * PI / 180;
}

fn isEqual(v1 : vec3<f32>, v2 : vec3<f32>) -> bool
{
    return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

@compute @workgroup_size(1,1,1)

fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

    let screenSize: vec2<u32> = textureDimensions(color_buffer);
    let screenPos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));
    
    /* World */
    setupWorld();

    /* Camera */
    setupCamera();

    var ray : Ray;
    let samplesPerPixel : u32 = 100;
    var pixel_color : vec3<f32> = vec3<f32>(0, 0, 0);
    for (var i : u32 = 0; i < samplesPerPixel; i++)
    {
        let u : f32 = (f32(screenPos.x) + rand()) / f32(screenSize.x - 1);
        let v : f32 = (f32(screenSize.y) - f32(screenPos.y) + rand()) / f32(screenSize.y - 1);

        let rd = worldCamera.lensRadius * randomInUnitDisk();
        let offset = worldCamera.u * rd.x + worldCamera.v * rd.y;
        ray.origin = worldCamera.origin + offset;
        ray.direction = worldCamera.lowerLeftCorner + u * worldCamera.horizontal + v * worldCamera.vertical - worldCamera.origin - offset;
        ray.time = randRange(0.0f, 1.0f);
        pixel_color += rayColor(ray);
    }
    pixel_color = pixel_color / f32(samplesPerPixel);
    pixel_color = sqrt(pixel_color);
    textureStore(color_buffer, screenPos, vec4<f32>(pixel_color, 1.0));
    //textureStore(color_buffer, screenPos, fin);


    
}

fn getUnitSphereUV(point : vec3<f32>) -> vec2<f32>
{
    let phi = atan2(-point.z, point.x) + PI;
    let theta = acos(-point.y);

    let u = phi / (2.0f * PI);
    let v = theta / PI;
    return vec2<f32>(u,v);
}

fn getShapePoint(shape : Shape, time : f32) -> vec3<f32>
{
    let dir = shape.point1 - shape.point;
    return shape.point + ((time - shape.time0) / (shape.time1 - shape.time0)) * dir;
    //return vec3<f32>(1.0, 0.0, -1.5);
}

fn mergeBoundingBoxes(box1 : Shape, box2 : Shape) -> Shape
{
    let minx = min(box1.min.x, box2.min.x);
    let miny = min(box1.min.y, box2.min.y);
    let minz = min(box1.min.z, box2.min.z);

    let maxx = max(box1.max.x, box2.max.x);
    let maxy = min(box1.max.y, box2.max.y);
    let maxz = min(box1.max.z, box2.max.z);

    var box : Shape;

    box.min = vec3<f32>(minx, miny, minz);
    box.max = vec3<f32>(maxx, maxy, maxz);
    box.shape = BOUNDINGBOX;
    box.material = 4;

    return box;

    
}

fn computeSphereBoundingBox(shape : Shape) -> Shape
{
    var boundingBox1 : Shape;
    var boundingBox2 : Shape;

    boundingBox1.min = shape.point - vec3<f32>(shape.dimension, shape.dimension, shape.dimension);
    boundingBox1.max = shape.point + vec3<f32>(shape.dimension, shape.dimension, shape.dimension);
    boundingBox1.shape = BOUNDINGBOX;
    boundingBox1.material = 4;

    boundingBox2.min = shape.point1 - vec3<f32>(shape.dimension, shape.dimension, shape.dimension);
    boundingBox2.max = shape.point1 + vec3<f32>(shape.dimension, shape.dimension, shape.dimension);
    boundingBox2.shape = BOUNDINGBOX;
    boundingBox2.material = 4;
    return mergeBoundingBoxes(boundingBox1, boundingBox2);
}

fn computeBoundingBox(shape : Shape) -> Shape
{
    var boundingBox : Shape;
    let shapeType = shape.shape;

    switch shapeType {
        case SPHERE: {
            boundingBox = computeSphereBoundingBox(shape);
            boundingBox.created = true;
        }
        default: {
            boundingBox.created = false;
        }
    }
    return boundingBox;
}

fn perlin_generate_perm() -> array<u32, 256>
{
    var point_count : array<u32, 256>;
    for (var x : u32 = 0; x < 256; x++)
    {
        point_count[x] = x;
    }
    for (var x : u32 = 255; x > 0; x--)
    {
        let t = randIntRange(0, i32(x));
        let temp = point_count[x];
        point_count[x] = point_count[t];
        point_count[t] = temp;
    }
    return point_count;
}

fn raySphereIntersection(ray : Ray, shape : Shape, tMin : f32, tMax: f32) -> HitRecord
{
    var hitRecord : HitRecord;
    var point : vec3<f32> = getShapePoint(shape, ray.time);
    var oc : vec3<f32> = ray.origin - point; //Vector from sphere center to ray origin
    var a  : f32 = lengthSquared(ray.direction);
    var half_b : f32 = dot(oc, ray.direction);
    var c  : f32 = lengthSquared(oc) - shape.dimension * shape.dimension;
    var discriminant  : f32 = half_b * half_b - a * c; // (0.5b)^2 - a * c

    //No intersection
    if(discriminant < 0)
    {
        hitRecord.hit = false;
        return hitRecord;
    }


    var sqrtd : f32 = sqrt(discriminant);
    var root : f32 = (-half_b - sqrtd) / a;
    if(root < tMin || root > tMax)
    {
        root = (-half_b + sqrtd) / a;
        if(root < tMin || root > tMax)
        {
            hitRecord.hit = false;
            return hitRecord;
        }
    }

    var surfacePoint : vec3<f32> = rayAt(ray, root);
    var normal : vec3<f32> = normalize(surfacePoint - point);

    //Assign hit record properties
    hitRecord.hit = true;
    hitRecord.t = root;
    hitRecord.point = surfacePoint;
    hitRecord.normal = faceForward(normal, normal, ray.direction);
    //hitRecord.normal = normal;
    hitRecord.frontFace = isEqual(normal, hitRecord.normal);
    if(shape.isTextured == 0)
    {
        hitRecord.albedo = shape.albedo;
    }
    else if(shape.isTextured == 2)
    {
        let uv = getUnitSphereUV(normal);
        let u = uv.x;
        let v = uv.y;
        let x = i32(1023 * u);
        let y = 511 - i32(511 * v);
        let value = texture[x + 1024 * y];
        let first = (value & 0x000000FFu) >> 0;
        let second = (value & 0x0000FF00u) >> 8;
        let third = (value & 0x00FF0000u) >> 16;
        let fourth = (value & 0xFF000000u) >> 24;
        let fin = vec4<f32>(
            f32(first) / 255.0,
            f32(second) / 255.0,
            f32(third) / 255.0,
            f32(fourth) / 255.0
        );
        hitRecord.albedo = fin.rgb;

    }
    else
    {
        let uv = getUnitSphereUV(normal);
        hitRecord.albedo = (1+ sin(2 * surfacePoint.y) + 5*turb(surfacePoint, 7)) * 0.5  * vec3<f32>(0.8);
        

        

    }
    
    hitRecord.material = shape.material;
    hitRecord.fuzziness = shape.fuzziness;
    hitRecord.etat = shape.etat;
    return hitRecord;
}

fn turb(surfacePoint : vec3<f32>, depth : i32) -> f32
{
    var acc : f32 = 0.0;
    var p : vec3<f32> = surfacePoint;
    var weight : f32 = 1.0;
    for(var x : i32 = 0; x < depth; x++)
    {
        acc = acc + weight*noise(p);
        weight = 0.5 * weight;
        p = 2.0 * p;
    }
    return abs(acc);
}



fn noise(surfacePoint : vec3<f32>) -> f32
{
    var u : f32 = surfacePoint.x - floor(surfacePoint.x);
        var v : f32 = surfacePoint.y - floor(surfacePoint.y);
        var w : f32  = surfacePoint.z - floor(surfacePoint.z);

        u = u*u*(3-2*u);
        v = v*v*(3-2*v);
        w = w*w*(3-2*w);

        let i = i32(floor(surfacePoint.x));
        let j = i32(floor(surfacePoint.y));
        let k = i32(floor(surfacePoint.z));

        
        
        var cell : array<array<array<vec3<f32>, 2>,2>, 2>;
    
        for(var di : i32 = 0; di < 2; di++)
        {
            for(var dj : i32 = 0; dj < 2; dj++)
            {
                for(var dk : i32 = 0; dk < 2; dk++)
                {
                    cell[di][dj][dk] = ranfloat[
                        perm_x[(i + di) & 255] ^
                        perm_y[(j + dj) & 255] ^
                        perm_z[(k + dk) & 255]
                    ];
                }
            }
        }
    var acc : f32 = 0.0;
        for (var a : i32 = 0; a < 2; a++)
        {
            for(var b : i32 =0; b < 2; b++)
            {
                for(var c : i32 = 0; c < 2; c++)
                {
                    let weight = vec3<f32>(u-f32(a), v-f32(b), w-f32(c));
                    acc = acc + (f32(a) * u + (1-f32(a)) * (1-u))*(f32(b) * v + (1-f32(b)) * (1 - v)) * (f32(c) * w + (1 - f32(c)) * (1 - w)) * dot(cell[a][b][c], weight);
                    //acc = acc + (f32(a) * su + (1-f32(a)) * (1-u))*(f32(b) * v + (1-f32(b)) * (1 - v)) * (f32(c) * w + (1 - f32(c)) * (1 - w)) * cell[a][b][c];
                }
            }
        }
        return acc;
}

fn rayPlaneIntersection(ray : Ray, shape : Shape, tMin : f32, tMax: f32) -> HitRecord
{
    var hitRecord : HitRecord;
    let normal = shape.normal;
    let denom = (dot(normal, ray.direction));
    let num = dot((shape.point - ray.origin), normal);
    let t = num/denom;
    if(t < tMin || t > tMax)
    {
        hitRecord.hit = false;
        return hitRecord;
    }

    hitRecord.hit = t >= 0;
    hitRecord.normal = normal;
    hitRecord.frontFace = isEqual(normal, hitRecord.normal);
    hitRecord.point = rayAt(ray, t);
    hitRecord.t = t;
    hitRecord.albedo = shape.albedo;
    hitRecord.material = shape.material;
    hitRecord.fuzziness = shape.fuzziness;
    hitRecord.etat = shape.etat;
    
    return hitRecord;
}

fn rayTriangleIntersection(ray : Ray, shape : Shape, tMin : f32, tMax: f32) -> HitRecord
{
    var hitRecord : HitRecord;

    let v1 = shape.v1;
    let v2 = shape.v2;
    let v3 = shape.v3;

    let e1 = v2 - v1;
    let e2 = v3 - v1;
    let p = cross(ray.direction, e2);
    let det = dot(e1, p);

    if (abs(det) < 0.0001 ) {
        hitRecord.hit = false;
        return hitRecord;
    }

    let invDet = 1.0 / det;
    let tvec = ray.origin - v1;
    let u = dot(tvec, p) * invDet;

    if (u < 0.0 || u > 1.0) {
        hitRecord.hit = false;
        return hitRecord;
    }

    let q = cross(tvec, e1);
    let v = dot(ray.direction, q) * invDet;
    
    if (v < 0.0 || u + v > 1.0) {
        hitRecord.hit = false;
        return hitRecord;
    }

    let t = dot(e2, q) * invDet;
    if (t > 0.0) {
        hitRecord.hit = true;
        hitRecord.material = 4;
        return hitRecord;
    }
    hitRecord.hit = false;
    return hitRecord;
}

fn rayBVHIntersection(ray : Ray, node : BVHNode, tMin : f32, tMax: f32) -> HitRecord
{
    var hitRecord : HitRecord;
    var stack : array<i32, MAX_STACK_SIZE>;
    var top : i32 = -1;
    top = top + 1;
    stack[top] = node.index;
    while(top >= 0)
    {
        let currentNodeIndex = stack[top];
        top = top - 1;

        let currentNode = nodes[currentNodeIndex];
        let currentBox = currentNode.box;
        hitRecord = rayBoundingBoxIntersection(ray, currentBox, tMin, tMax);
        if(hitRecord.hit)
        {
            if(currentNode.right == -1 && currentNode.left == -1)
            {
                return rayIntersection(ray , currentNode.shape, tMin , tMax);
            }
            if(currentNode.right != -1)
            {
                top = top + 1;
                stack[top] = currentNode.right;
            }
            if(currentNode.left != -1)
            {
                top = top + 1;
                stack[top] = currentNode.left;
            }
        }
    }
    hitRecord.hit = false;
    return hitRecord;
}

fn rayBoundingBoxIntersection(ray : Ray, shape : Shape, tMin : f32, tMax: f32) -> HitRecord
{
    var hitRecord : HitRecord;
    var t_min = tMin;
    var t_max = tMax;
    var t : f32 = 0;
    //var zcorrection : f32 = 1;
    var flipped : bool = false;
    var instanceFlipped : bool = false;
    for (var x: u32 = 0; x < 3; x++)
    {
        if(x == 2)
        {
            //zcorrection = -1;
        }
        let invD = 1.0 / ray.direction[x];
        var t0 = (shape.min[x] - ray.origin[x]) * invD;
        var t1 = (shape.max[x] - ray.origin[x]) * invD;
        if(invD < 0.0f)
        {
            let temp = t0;
            t0 = t1;
            t1 = temp;
            instanceFlipped = true;
        }else {
            instanceFlipped = false;
        }
        if(t0 > t_min)
        {
            t_min = t0;
            flipped = instanceFlipped;
        }
        if(t1 < t_max)
        {
            t_max = t1;
        }
        if(t_min >= t_max)
        {
            hitRecord.hit = false;
            return hitRecord;
        }
    }
    
    hitRecord.hit = true;
    hitRecord.t = t_min;
    hitRecord.point = rayAt(ray, t_min);
    hitRecord.material = shape.material;


    var xdiff : f32;
    var ydiff : f32;
    var zdiff : f32;

    if(flipped)
    {
        xdiff = abs(hitRecord.point.x - shape.max.x) / abs(shape.max.x - shape.min.x);
        ydiff = abs(hitRecord.point.y - shape.max.y) / abs(shape.max.y - shape.min.y);
        zdiff = abs(hitRecord.point.z - shape.max.z) / abs(shape.max.z - shape.min.z);
    }else
    {
        /*
            blueBoundingBox.min = vec3<f32>(-2.5, 0.5, -1.0);
            blueBoundingBox.max = vec3<f32>(1.5, 1.5, -2.0);
        */
        xdiff = abs(hitRecord.point.x - shape.min.x) / abs(shape.max.x - shape.min.x);
        ydiff = abs(hitRecord.point.y - shape.min.y) / abs(shape.max.y - shape.min.y);
        zdiff = abs(hitRecord.point.z - shape.min.z) / abs(shape.max.z - shape.min.z);
    }

    

    var normal : vec3<f32>;
    hitRecord.albedo = shape.albedo;

    if(xdiff < ydiff && xdiff < zdiff)
    {
        if(flipped)
        {
            normal = vec3<f32>(1.0, 0.0, 0.0);
        }
        else{
            normal = vec3<f32>(-1.0, 0.0, 0.0);
        }
    }

    if(ydiff < zdiff && ydiff < xdiff)
    {
        if(flipped)
        {
            normal = vec3<f32>(0.0, 1.0, 0.0);
        }
        else{
            normal = vec3<f32>(0.0, -1.0, 0.0);
        }
    }

    if(zdiff < ydiff && zdiff < xdiff)
    {
        if(flipped)
        {
            normal = vec3<f32>(0.0, 0.0, -1.0);
        }
        else{
            normal = vec3<f32>(0.0, 0.0, 1.0);
        }
    }

    normal = vec3<f32>(0.0, 0.0, 1.0);
    hitRecord.normal = normal;
    
    return hitRecord;
}

/* Ray-shape intersection tests */
fn rayIntersection(ray : Ray, shape : Shape, tMin : f32, tMax: f32) -> HitRecord 
{
    var hitRecord : HitRecord;
    let shapeType = shape.shape;

    switch shapeType {
        case SPHERE: {
            hitRecord = raySphereIntersection(ray, shape, tMin, tMax);
        }
        case PLANE: {
            hitRecord = rayPlaneIntersection(ray, shape, tMin, tMax);
        }
        case TRIANGLE: {
            hitRecord = rayTriangleIntersection(ray, shape, tMin, tMax);
        }
        case BOUNDINGBOX: {
            hitRecord = rayBoundingBoxIntersection(ray, shape, tMin, tMax);
        }   
        default {
            hitRecord.hit = false;
        }
    }
    return hitRecord;
    
}

fn rayAt(ray : Ray, t : f32) -> vec3<f32>
{
    return ray.origin + t * ray.direction;
}

fn rayColor(ray : Ray) -> vec3<f32>
{
    let depth : u32 = 50;
    var color = vec3<f32>(0,0,0);
    let grad = normalize(ray.direction).y;
    var localRay : Ray;
    localRay.origin = ray.origin;
    localRay.direction = ray.direction;
    localRay.time = ray.time;
    var attuenation : vec3<f32> = vec3<f32>(1.0);
    var mask : i32 = -1;
    var carryMask : i32 = -1;
    for (var x : u32 = 0; x < depth; x++)
    {
        var hitAnything : bool = false;
        var nearest : f32 = INFINITY;
        var hitRecord : HitRecord;
        var temp : HitRecord;
        if(isMeshLoaded == 1)
        {
            for (var i: u32 = 0; i < numOfTriangles; i++) 
            {
            
                var tri : Shape;
                tri.shape = TRIANGLE;
                tri.v1 = vertices[triangles[i].x];
                tri.v2 = vertices[triangles[i].y];
                tri.v3 = vertices[triangles[i].z];
                tri.material = 4;
                temp = rayIntersection(localRay, tri, 0.001, nearest);
                if(temp.hit)
                {
                    hitAnything = true;
                    nearest = temp.t;
                    hitRecord = temp;
                    break;
                }
            }
        } else 
        {
            for (var i: u32 = 0; i < NUM_OF_OBJECTS; i++) 
            {
                temp = rayIntersection(localRay, objects[i], 0.001, nearest);
                if(temp.hit)
                {
                    hitAnything = true;
                    nearest = temp.t;
                    hitRecord = temp;
                    break;
                }
                
            }
        }
        

        if(hitAnything)
        {
            localRay.origin = hitRecord.point;
            if(hitRecord.material == 1)
            {

                localRay.direction = hitRecord.normal + randomUnitVector();
                if(nearZero(localRay.direction))
                {
                    localRay.direction = hitRecord.normal;
                }
            }
            if(hitRecord.material == 2)
            {

                localRay.direction = reflect(normalize(localRay.direction), hitRecord.normal) + hitRecord.fuzziness * randomInUnitSphere();
                if(dot(localRay.direction, hitRecord.normal) < 0)
                {
                    break;
                }
            }
            if(hitRecord.material == 3)
            {
                let etaiOverEtat = select(hitRecord.etat, 1.0/hitRecord.etat, hitRecord.frontFace);
                let unitDirection = normalize(localRay.direction);
                let cosTheta = min(dot(-unitDirection, hitRecord.normal), 1.0);
                localRay.direction = refract(unitDirection, hitRecord.normal, etaiOverEtat);
                if(isEqual(localRay.direction, vec3<f32>(0)) || reflectance(cosTheta, etaiOverEtat) > rand())
                {

                    localRay.direction = reflect(normalize(localRay.direction), hitRecord.normal);
                }
            }
            if(hitRecord.material == 4)
            {
                localRay.direction = hitRecord.normal + randomUnitVector();
                if(nearZero(localRay.direction))
                {
                    localRay.direction = hitRecord.normal;
                }
                
            }

            attuenation *= hitRecord.albedo;
            //attuenation = vec3<f32>(min(attuenation.x, hitRecord.albedo.x),min(attuenation.y, hitRecord.albedo.y),min(attuenation.z, hitRecord.albedo.z))
        }
        else
        {
            break;
        }

    }
    
    


    let t2 = 0.5 * (grad + 1.0);

        color += attuenation * ((1.0 - t2) * vec3<f32>(1.0, 1.0, 1.0) + t2 * vec3<f32>(0.5, 0.7, 1.0));

    
    return color;
}





/* Functions to create a PRNG on the GPU */
fn tea(val0:u32, val1:u32)->u32{
// "GPU Random Numbers via the Tiny Encryption Algorithm"
  var v0 = val0;
  var v1 = val1;
  var s0 = u32(0);
  for (var n: i32 = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}
fn rand()->f32{
  // Generate a random float in [0, 1) 
  return (f32(lcg()) / f32(0x01000000));
}
fn lcg()->u32{
// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
  let LCG_A = 1664525u;
  let LCG_C = 1013904223u;
  seed = (LCG_A * seed + LCG_C);
  return seed & 0x00FFFFFF;
}

fn randIntRange(min : i32, max : i32) -> i32
{
    let minf = f32(min);
    let maxf = f32(max);
    let num =  minf + rand() * (maxf - minf);
    return i32(num);
}

fn randRange(min : f32, max : f32) -> f32
{
    return min + (max - min) * rand();
}

fn randVector() -> vec3<f32> 
{
    return vec3<f32>(rand(), rand(), rand());
}

fn randVectorRange(min : f32, max : f32) -> vec3<f32>
{
    return vec3<f32>(randRange(min, max), randRange(min, max), randRange(min, max));
}

fn randomInUnitSphere() -> vec3<f32>
{
    while(true)
    {
        let v = randVectorRange(-1, 1);
        if(lengthSquared(v) >= 1)
        {
            continue;
        }
        return v;
    }
    return vec3<f32>(0,0,0);
}

fn randomUnitVector() -> vec3<f32>
{
    return normalize(randomInUnitSphere());
}

fn nearZero(v : vec3<f32>) -> bool
{
    let epsilon = 1e-8;
    return abs(v.x) < epsilon && abs(v.y) < epsilon && abs(v.z) < epsilon;
}

fn reflectance(cosine : f32, etaiOverEtat : f32) -> f32
{
    let r0 = (1-etaiOverEtat) / (1+etaiOverEtat);
    let r02 = r0 * r0;
    return r02 + (1-r0) * pow((1-cosine), 5);
}

fn randomInUnitDisk() -> vec3<f32>
{
    while(true)
    {
        let v = vec3<f32>(randRange(-1,1),randRange(-1,1),0);
        if(lengthSquared(v) >= 1)
        {
            continue;
        }
        return v;
    }
    return vec3<f32>(0);
}

var<private> seed: u32 = 0;