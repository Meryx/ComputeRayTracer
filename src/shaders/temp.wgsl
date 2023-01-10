@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> camera: array<f32, 11>;
@group(0) @binding(2) var<uniform> view : mat4x4<f32>;

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

var<private> seed: u32 = 0;

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct Plane {
    point: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
}

struct Circle { 
    point: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    radius: f32,
    transform: mat4x4<f32>,
    area: f32,
}

struct hitRecord {
    surfacePoint: vec3<f32>,
    normal: vec3<f32>,
    hit: bool,
    t: f32,
    albedo: vec3<f32>,
}

fn rayAt(ray: Ray, t: f32) -> vec3<f32>
{
    return ray.origin + t * ray.direction;
}

fn plane_hit(ray: Ray, plane: Plane) -> hitRecord {
    var hitRecord: hitRecord;
    let denom = (dot(plane.normal, ray.direction));
    // if(denom < 1e-6)
    // {
    //     hitRecord.hit = false;
    //     return hitRecord;
    // }
    let num = dot((plane.point - ray.origin), plane.normal);
    let t = num/denom;
    hitRecord.hit = t >= 0;
    hitRecord.normal = plane.normal;
    hitRecord.surfacePoint = rayAt(ray, t);
    hitRecord.t = t;
    hitRecord.albedo = plane.albedo;
    return hitRecord;

}


fn circle_hit(ray: Ray, circle: Circle) -> hitRecord {
    var hitRecord: hitRecord;
    let denom = (dot(circle.normal, ray.direction));
    // if(denom < 1e-6)
    // {
    //     hitRecord.hit = false;
    //     return hitRecord;
    // }
    let num = dot((circle.point - ray.origin), circle.normal);
    let t = num/denom;
    hitRecord.surfacePoint = rayAt(ray, t);
    hitRecord.hit = t >= 0 && length(hitRecord.surfacePoint - circle.point) <= circle.radius;
    hitRecord.normal = circle.normal;
    hitRecord.t = t;
    hitRecord.albedo = circle.albedo;
    return hitRecord;

}

fn sample_circle() -> vec2<f32> {
    let x0 = 2 * rand() - 1;
    let x1 = 2 * rand() - 1;
    if(x0 == 0 && x1 == 0)
    {
        return vec2<f32>(0, 0);
    }
    if(abs(x0) > abs(x1))
    {
        let r = x0;
        let theta = (PI/4) * (x1/x0);
        return r * vec2<f32>(cos(theta), sin(theta));
    }
    else
    {
        let r = x1;
        let theta = (PI/2) - (PI/4) * (x0/x1);
        return r * vec2<f32>(cos(theta), sin(theta));
    }
}

fn random_point_circle(circle : Circle) -> vec3<f32> {
    let pd = sample_circle();
    let pObj = circle.radius * pd;
    let unit : vec4<f32> = vec4<f32>(pObj.x, pObj.y, 0, 1);
    let f = circle.transform * unit;
    return f.xyz;
}

fn hit(ray: Ray, sphere: Sphere) -> hitRecord {
    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = 2.0 * dot(ray.direction, ray.origin - sphere.center);
    let c: f32 = dot(ray.origin - sphere.center, ray.origin - sphere.center) - sphere.radius * sphere.radius;
    let discriminant: f32 = b * b - 4.0 * a * c;
    var hitRecord: hitRecord;
    if(discriminant < 0)
    {
        hitRecord.hit = false;
        return hitRecord;
    }
    let t1 = (-b + sqrt(discriminant)) / 2.0 * a;
    let t2 = (-b - sqrt(discriminant)) / 2.0 * a;
    let t = min(t1, t2);
    let surfacePoint = rayAt(ray, t);
    let normal = normalize(surfacePoint - sphere.center);
    hitRecord.hit = true;
    hitRecord.surfacePoint = surfacePoint;
    hitRecord.normal = normal;
    hitRecord.t = t;
    hitRecord.albedo = sphere.albedo;
    return hitRecord;
}

const PI: f32 = 3.14159265359;
const roughness: f32 = 0.4;
const metallic: f32 = 0.1;
// const albedo: vec3<f32> = vec3<f32>(0.7, 0.1, 1.0);

@compute @workgroup_size(1,1,1)

fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

    let screen_size: vec2<u32> = textureDimensions(color_buffer);
    let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));
    let horizontal_coefficient: f32 = (f32(screen_pos.x) + 0.5) / f32(screen_size.x);
    let vertical_coefficient: f32 = (f32(screen_size.y) - f32(screen_pos.y) + 0.5) / f32(screen_size.y);


    var mySphere: Sphere;
    mySphere.center = vec3<f32>(0.0, 0.0, -800);
    mySphere.radius = 320;
    mySphere.albedo = vec3<f32>(0.7, 0.1, 1.0);

    var myPlane: Plane;
    myPlane.point = vec3<f32>(0.0, -1000.0, 0.0);
    myPlane.normal = vec3<f32>(0.0, 1.0, 0.0);
    myPlane.albedo = vec3<f32>(0.6, 0.6, 0.6);

    var myPlane2: Plane;
    myPlane2.point = vec3<f32>(-1000, 0.0, 0.0);
    myPlane2.normal = vec3<f32>(1.0, 0.0, 0.0);
    myPlane2.albedo = vec3<f32>(0.9, 0.1, 0.1);

    var myPlane3: Plane;
    myPlane3.point = vec3<f32>(0.0, 0.0, -1500);
    myPlane3.normal = vec3<f32>(0.0, 0.0, 1.0);
    myPlane3.albedo = vec3<f32>(0.1, 0.9, 0.1);

    var myPlane4: Plane;
    myPlane4.point = vec3<f32>(1000, 0.0, 0.0);
    myPlane4.normal = vec3<f32>(-1.0, 0.0, 0.0);
    myPlane4.albedo = vec3<f32>(0.1, 0.1, 0.9);

    var myPlane5: Plane;
    myPlane5.point = vec3<f32>(0.0, 1200, 0.0);
    myPlane5.normal = vec3<f32>(0.0, -1.0, 0.0);
    myPlane5.albedo = vec3<f32>(0.6, 0.6, 0.6);

    var myPlane6: Plane;
    myPlane6.point = vec3<f32>(0.0, 0.0, 1500);
    myPlane6.normal = vec3<f32>(0.0, 0.0, -1.0);
    myPlane6.albedo = vec3<f32>(0.1, 0.9, 0.1);

    var myCircle: Circle;
    myCircle.point = vec3<f32>(0.0, 1200, -800);
    myCircle.normal = vec3<f32>(0.0, -1.0, 0.0);
    myCircle.albedo = vec3<f32>(1, 1, 1);
    myCircle.radius = 320;
    myCircle.area = myCircle.radius * myCircle.radius * PI;

    let translation = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(myCircle.point.xyz, 1.0));
    let rotation = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, cos(PI/2), sin(PI/2), 0.0), vec4<f32>(0.0, -sin(PI/2), cos(PI/2), 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let transform = translation * rotation;
    myCircle.transform = transform;

    let cameraUp: vec3<f32> = vec3<f32>(camera[6], camera[7], camera[8]);
    let w: vec3<f32> = normalize(vec3<f32>(-camera[3], -camera[4], -camera[5]));
    let u: vec3<f32> = normalize(cross(cameraUp, w));
    let v: vec3<f32> = cross(w, u);
    let cameraOrigin: vec3<f32> = vec3<f32>(camera[0], camera[1], camera[2]);
    let focalDistance: f32 = camera[9];
    let aspectRatio: f32 = camera[10];


    var myRay: Ray;

    var rayOrigin: vec4<f32> = vec4<f32>(0,0,0,1);
    var rayDirection: vec4<f32> = vec4<f32>(0,0,-1,0);
    rayOrigin = view * rayOrigin;
    rayDirection = view * rayDirection;
        



    let forwards = rayDirection.xyz;
    let left = aspectRatio * (view * vec4<f32>(-1,0,0,0)).xyz;
    let right = 2 * (-left);
    let down = (view * vec4<f32>(0,-1,0,0)).xyz;
    let up = 2 * (-down);

    myRay.origin = rayOrigin.xyz;
    myRay.direction = normalize(forwards + left + horizontal_coefficient * right  + down + (vertical_coefficient * up));
    


    let lightPosition: vec3<f32> = vec3<f32>(0, 1200, -800);
    let lightColor: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

    var pixel_color : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var tempT : hitRecord;
    var hitRecord = hit(myRay, mySphere);
    tempT = plane_hit(myRay, myPlane);
    if(!hitRecord.hit || (tempT.hit && tempT.t < hitRecord.t))
    {
        hitRecord = tempT;
    }

    tempT = plane_hit(myRay, myPlane2);
    if(!hitRecord.hit || (tempT.hit && tempT.t < hitRecord.t))
    {
        hitRecord = tempT;
    }

    tempT = plane_hit(myRay, myPlane3);
    if(!hitRecord.hit || (tempT.hit && tempT.t < hitRecord.t))
    {
        hitRecord = tempT;
    }

    tempT = plane_hit(myRay, myPlane4);
    if(!hitRecord.hit || (tempT.hit && tempT.t < hitRecord.t))
    {
        hitRecord = tempT;
    }

    tempT = plane_hit(myRay, myPlane5);
    if(!hitRecord.hit || (tempT.hit && tempT.t < hitRecord.t))
    {
        hitRecord = tempT;
    }

    tempT = plane_hit(myRay, myPlane6);
    if(!hitRecord.hit || (tempT.hit && tempT.t < hitRecord.t))
    {
        hitRecord = tempT;
    }

    tempT = circle_hit(myRay, myCircle);
    if(!hitRecord.hit || (tempT.hit && tempT.t <= hitRecord.t))
    {
        hitRecord = tempT;
    }
    var accm : vec3<f32> = vec3<f32>(0,0,0);
    var hit_light : bool = false;
    if (hitRecord.hit) {

            let albedo = hitRecord.albedo;
            let surfacePoint = hitRecord.surfacePoint;
            let normal = hitRecord.normal;


        
            let orig = hitRecord.surfacePoint;
            for (var i: i32 = 0; i < 100; i++) {
                
                if(length(orig - myCircle.point) <= myCircle.radius)
                {
                    accm = vec3<f32>(1.0, 1.0, 1.0);
                    hit_light = true;
                    break;
                }




                let rand = random_point_circle(myCircle);
                let saber = (rand - orig);
                let dir = normalize(saber);
                // let saber2 = pow(length(saber),2);
                let saber2 = dot(saber, saber);
                var lightRay : Ray;
                lightRay.origin = orig;
                lightRay.direction = dir;
                if(dot(dir, myCircle.normal) >= 0)
                {
                    break;
                }
                // tempT = circle_hit(lightRay, myCircle);
                let pdf = saber2 / (abs(dot(-dir, myCircle.normal)) * myCircle.area);





                let l = normalize(rand - surfacePoint);
                let v = normalize(myRay.origin - surfacePoint);
                let h = normalize(v + l);
                let distance = length(rand - surfacePoint);
                // let attuenation = 1.0 / (distance * distance);
                let attuenation = 1.0;
                let radiance = lightColor * attuenation;
                let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
                let cosTheta = max(dot(h,v), 0.0);
                let schlick = f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);

                let a = roughness * roughness;
                let a2 = a * a;
                let ndoth = max(dot(normal, h), 0.0);
                let ndoth2 = ndoth * ndoth;

                let term = (ndoth2 * (a2 - 1.0) + 1.0);
                let denom = PI * term * term;
                let dggx = a2 / denom;

                let r = (roughness + 1.0);
                let k = (r * r) / 8.0;
                let ndotv = max(dot(normal, v), 0.0);
                let denom2 = ndotv * (1.0 - k) + k;
                let geoggx = ndotv / denom2;

                let ndotl = max(dot(normal, l), 0.0);
                let geoggx2 = ndotl / (ndotl * (1.0 - k) + k);
                let smith = geoggx * geoggx2;

                let ndf = dggx;
                let g = smith;
                let numerator = ndf * g * schlick;
                let denominator = 4.0 * ndotv * ndotl  + 0.0001;
                let spec = numerator / denominator;

                let ks = (schlick);
                let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);
                let lo1 = (kd * albedo / PI + spec) * radiance * ndotl;
                let lo = lo1/pdf;
                accm = accm + lo;






            }
            let amb = vec3<f32>(0.01) * albedo;
            accm = accm/100;
            accm = accm + amb;
            let temp = accm / (accm + vec3<f32>(1.0));
            let color = pow(temp,vec3(1.0/2.2));
            pixel_color = color;
            if(hit_light)
            {
                pixel_color = vec3<f32>(1,1,1);
            }

        





            
        

    }

    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.0));
}