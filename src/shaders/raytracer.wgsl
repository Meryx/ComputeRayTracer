@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> isMeshLoaded: u32;
@group(0) @binding(2) var<storage, read_write> numOfTriangles: u32;
@group(0) @binding(3) var<storage> triangles: array<vec3<u32>>;
@group(0) @binding(4) var<storage> vertices: array<vec3<f32>>;


/* Numerical constants */
const PI: f32 = 3.14159265359;
const INFINITY : f32 = 3.40282346638528859812e+38f;
const NUM_OF_OBJECTS : u32 = 2;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}


const SPHERE : u32 = 1;
const PLANE : u32 = 2;
const TRIANGLE : u32 = 3;

struct Shape {
    shape: u32,
    point: vec3<f32>,
    albedo: vec3<f32>,
    dimension: f32,
    material: u32,
    fuzziness: f32,
    etat: f32,
    normal: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    v3: vec3<f32>,
}

struct HitRecord { 
    point: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
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
}



fn setupCamera() 
{

    let focalDistance: f32 = 1.0;
    let aspectRatio: f32 = 3.0 / 2.0;
    let fov : f32 = 90;
    let theta = degreesToRadians(fov);
    let h = tan(theta/2);
    let viewportHeight = 2.0 * h;
    let viewportWidth = aspectRatio * viewportHeight;

    let lookFrom = vec3<f32>(0,0,1);
    let lookAt = vec3<f32>(0,0,0);
    let aperture = 0.1;
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
}

var<private> objects : array<Shape, NUM_OF_OBJECTS>;
var<private> worldCamera : Camera;
var<private> redSphere : Shape;
var<private> grayPlane : Shape;

fn setupWorld() {
    redSphere.shape = SPHERE;
    redSphere.point = vec3<f32>(0, 0, -1.5);;
    redSphere.dimension = 0.7;
    
    redSphere.material = 1;

    redSphere.albedo = vec3<f32>(0.8, 0.0, 0.0);
    redSphere.albedo = vec3<f32>(0.0, 0.8, 0.0);
    objects[0] = redSphere;

    grayPlane.shape = PLANE;
    grayPlane.point = vec3<f32>(0.0f, -0.75f, 0.0f);
    grayPlane.albedo = vec3<f32>(0.7, 0.7, 0.7);
    grayPlane.material = 2;
    grayPlane.normal = vec3<f32>(0.0f, 1.0f, 0.0f);
    grayPlane.fuzziness = 0.1;
    objects[1] = grayPlane;

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
        pixel_color += rayColor(ray);
    }
    pixel_color = pixel_color / f32(samplesPerPixel);
    textureStore(color_buffer, screenPos, vec4<f32>(pixel_color, 1.0));

    
}

fn raySphereIntersection(ray : Ray, shape : Shape, tMin : f32, tMax: f32) -> HitRecord
{
    var hitRecord : HitRecord;
    let oc = ray.origin - shape.point; //Vector from sphere center to ray origin
    let a = lengthSquared(ray.direction);
    let half_b = dot(oc, ray.direction);
    let c = lengthSquared(oc) - shape.dimension * shape.dimension;
    let discriminant = half_b * half_b - a * c; // (0.5b)^2 - a * c

    //No intersection
    if(discriminant < 0)
    {
        hitRecord.hit = false;
        return hitRecord;
    }


    let sqrtd = sqrt(discriminant);
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

    let surfacePoint = rayAt(ray, root);
    let normal = normalize(hitRecord.point - shape.point);

    //Assign hit record properties
    hitRecord.hit = true;
    hitRecord.t = root;
    hitRecord.point = rayAt(ray, root);
    hitRecord.normal = faceForward(normal, normal, ray.direction);
    hitRecord.frontFace = isEqual(normal, hitRecord.normal);
    hitRecord.albedo = shape.albedo;
    hitRecord.material = shape.material;
    hitRecord.fuzziness = shape.fuzziness;
    hitRecord.etat = shape.etat;
    return hitRecord;
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
    var attuenation : vec3<f32> = vec3<f32>(1.0);
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
                attuenation = vec3<f32>(0.0, 1.0, 0.0);
                break;
            }

            attuenation *= hitRecord.albedo;
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