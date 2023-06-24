@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage> spheres : array<Sphere>;
@group(0) @binding(2) var<storage, read_write> alt_color_buffer : array<vec3<f32>>;
@group(0) @binding(3) var<uniform> sample : u32;
@group(0) @binding(4) var<storage> planar_patches : array<PlanarPatch>;
@group(0) @binding(5) var<storage> camera : Camera;


const PI: f32 = 3.14159265359;
const MAXDEPTH : u32 = 1000000000;
const INFINITY : f32 = 3.40282346638528859812e+38f;
const GRID_SIZE : u32 = 16;
const BRDF : f32 = 1.0 / PI;
const lambda_min : f32 = 400.0;
const lambda_max : f32 = 700.0;
const light_spectrum : vec4<f32> = vec4<f32>(0, 8.0, 15.6, 18.4);
const light_reflectance : f32 = 0.78;


    
struct Sphere {
  center: vec3<f32>,
  radius: f32,
  albedo: vec3<f32>,
  index: u32,
  emission: vec3<f32>
};

struct Camera {
  eye: vec3<f32>,
  lookat: vec3<f32>,
  up: vec3<f32>,
  viewport_width: f32,
  viewport_height: f32,
  focal_length: f32
};

struct PlanarPatch {
  origin: vec3<f32>,
  edge1: vec3<f32>,
  edge2: vec3<f32>,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  index: u32
};

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>
};

struct HitRecord {
  p: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  hit: bool,
  index: u32,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  last_index: u32,
};

struct ShapeIntersection {
  position: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  hit: bool,
  index: u32,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  last_index: u32,
};

fn camera_basis() -> mat3x3<f32> {
  let w : vec3<f32> = normalize(camera.eye - camera.lookat);
  let u : vec3<f32> = normalize(cross(camera.up, w));
  let v : vec3<f32> = cross(w, u);
  return mat3x3<f32>(u, v, w);
}

fn camera_ray(screen_pos : vec2<u32>, screen_size : vec2<u32>) -> Ray {
  let aspect_ratio : f32 = camera.viewport_width / camera.viewport_height;
  let viewport_height : f32 = 2.0 * tan(camera.focal_length / 2.0);
  let viewport_width : f32 = aspect_ratio * viewport_height;
  let basis = camera_basis();
  let u : vec3<f32> = basis[0];
  let v : vec3<f32> = basis[1];
  let w : vec3<f32> = basis[2];
  let horizontal : vec3<f32> = viewport_width * u;
  let vertical : vec3<f32> = viewport_height * v;
  let lower_left_corner : vec3<f32> = camera.eye - horizontal / 2.0 - vertical / 2.0 - w;

  let camera_film_parameters : vec2<f32> = camera_film_parameters(screen_pos, screen_size);
  let s = camera_film_parameters.x;
  let t = camera_film_parameters.y;

  return Ray(camera.eye, normalize(lower_left_corner + s * horizontal + t * vertical - camera.eye));
}

fn camera_film_parameters(screen_pos : vec2<u32>, screen_size : vec2<u32>) -> vec2<f32> {
  let s : f32 = (f32(screen_pos.x) + (f32(sample % GRID_SIZE) + rand()) / f32(GRID_SIZE)) / f32(screen_size.x);
  let t : f32 = (f32(screen_size.y) - f32(screen_pos.y) + (f32(sample % GRID_SIZE) + rand()) / f32(GRID_SIZE)) / f32(screen_size.y);
  return vec2<f32>(s,t);
}

fn path_trace(input_ray : Ray) -> vec3<f32>
{
  var ray : Ray = input_ray;
  var depth : u32 = 0;
  var radiance : vec4<f32> = vec4<f32>(0.0);
  var beta : vec4<f32> = vec4<f32>(1.0);
  var pdf_b : f32 = 1.0;
  var exclude : u32 = 100;
  var prev_intersection : ShapeIntersection;
  let sampled_spec : array<vec4<f32>, 2> = sample_light_spectrum();
  let lambdas = vec4<u32>(u32(sampled_spec[1].x), u32(sampled_spec[1].y), u32(sampled_spec[1].z), u32(sampled_spec[1].w));
  let spec = sampled_spec[0];
  let white_spec = sample_piecewise_linear_white(lambdas);
  let red_spec = sample_piecewise_linear_red(lambdas);
  let green_spec = sample_piecewise_linear_green(lambdas);

  while(true)
  {
    let shape_intersection : ShapeIntersection = intersect(ray, exclude);
    if(!shape_intersection.hit)
    {
      break;
    }
    exclude = shape_intersection.index;


    var le = shape_intersection.emission;
    if(length(le) > 0)
    {
      var le2 = spec;
      if(depth == 0)
      {
        radiance += beta * le2;
      } else {
        let light = planar_patches[shape_intersection.index];
        let area = length(light.edge1) * length(light.edge2);
        let abs_cos_theta_l = abs(dot(shape_intersection.normal, -ray.direction));
        let distance = length(prev_intersection.position - shape_intersection.position);
        let distance_squared = pow(distance, 2);
        let pdf_l_area = 1.0 / area;
        let geometric_term = abs_cos_theta_l / distance_squared;
        let pdf_l = pdf_l_area / geometric_term;
        let weight_b = power_heuristic(1, pdf_b, 1, pdf_l);
        radiance += weight_b * le2 * beta;
      }
      break;
    }

    if(depth >= MAXDEPTH)
    {
      break;
    }

    if(shape_intersection.albedo.x < 0.748 && shape_intersection.albedo.x > 0.746)
    {
      beta *= white_spec;
    }

    if(shape_intersection.albedo.x == 1)
    {
      beta *= red_spec;
    }

    if(shape_intersection.albedo.x == 2)
    {
      beta *= green_spec;
    }

    //beta *= shape_intersection.albedo;

    let light = planar_patches[2];
    var normal = normalize(cross((light.edge1), (light.edge2)));
    let ndotd = dot(normal, ray.direction);
    if(ndotd > 0)
    {
      normal = -normal;
    }
    let light_normal = normal;
    let area = length(light.edge1) * length(light.edge2);
    let point_on_light = sample_light();
    let is_visible = is_visible(shape_intersection.position, point_on_light, exclude);
    let light_dir = normalize(point_on_light - shape_intersection.position);
    let cos_theta_i = max(0, dot(shape_intersection.normal, light_dir));

    let radiance_light = spec * cos_theta_i;

    let abs_cos_theta_l = max(0.000001, abs(dot(light_normal, -light_dir)));
    let distance = length(point_on_light - shape_intersection.position);
    let distance_squared = pow(distance, 2);
    let pdf_l_area = 1.0 / area;
    let geometric_term = abs_cos_theta_l / distance_squared;
    
    let pdf_l = pdf_l_area / geometric_term;

    pdf_b = cos_theta_i / PI;
    let weight_l = power_heuristic(1, pdf_l, 1, pdf_b);
    
    radiance += BRDF * (is_visible * radiance_light) * weight_l * beta / pdf_l;

    let new_direction_with_pdf = cosine_weighted_sample_hemisphere(shape_intersection.normal);
    let new_direction = new_direction_with_pdf.xyz;
    pdf_b = new_direction_with_pdf.w;

    let cosnew = abs(dot(shape_intersection.normal, new_direction));

    beta *= BRDF * cosnew / (pdf_b);

    let max_beta_component = max(beta.x, max(beta.y, beta.z));
    if(depth > 1 && max_beta_component < 1)
    {
      let q = max(0, 1 - max_beta_component);
      if(rand() < q)
      {
        break;
      }
      beta /= 1 - q;
    }

    prev_intersection = shape_intersection;

    ray.origin = shape_intersection.position;
    ray.direction = new_direction;
    depth++;
  }
  var r = spectral_to_xyz(radiance, lambdas);
  r =  white_balance(r, spec, lambdas);
  return r;
}

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {


    /* Intialize screen size and position. Return if outside bounds of framebuffer. */
    let screen_size: vec2<u32> = vec2<u32>(u32(camera.viewport_width), u32(camera.viewport_height));
    let screen_pos : vec2<u32> = vec2<u32>(u32(GlobalInvocationID.x), u32(GlobalInvocationID.y));
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }

    /* Initialize random seed. Multiply x by 100 and y by 100 to ensure unique values between every pixel */
    seed = vec4<u32>(screen_pos.y, screen_pos.x * 100, sample, tea(screen_pos.x, screen_pos.y * 100));



    var ray : Ray = camera_ray(screen_pos, screen_size);
    var c : vec3<f32> = path_trace(ray);
    
    let pixel_index = screen_pos.x + screen_pos.y * screen_size.x;

    var xyzc = c;


    alt_color_buffer[pixel_index] += xyzc;
    xyzc = alt_color_buffer[pixel_index];
    xyzc = xyzc / f32(sample);

    xyzc = xyz_to_rgb(xyzc);


   

    //xyzc = xyzc / (xyzc + vec3<f32>(1.0));
    //xyzc = pow(xyzc, vec3<f32>(1.0 / 2.2));

    textureStore(framebuffer, screen_pos, vec4<f32>(xyzc, 1.0));
    
}

fn white_balance(xyz : vec3<f32> , source_spec : vec4<f32>, lambdas : vec4<u32>) -> vec3<f32>
{
  let M_bradford : mat3x3<f32> = mat3x3<f32>(0.8951,  0.2664, -0.1614,
        -0.7502,  1.7135,  0.0367,
         0.0389, -0.0685,  1.0296);

  let M_bradford_inv : mat3x3<f32> = mat3x3<f32>(0.9869929, -0.1470543,  0.1599627,
        0.4323053,  0.5183603,  0.0492912,
       -0.0085287,  0.0400428,  0.9684867);

  let source_illuminant_xyz = spectral_to_xyz_m();
  //let source_illuminant_xyz = vec3<f32>(0.7, 1.1, 0.5);
  let target_illuminant_xyz = vec3<f32>(0.95047, 1.0, 1.08883);
  
  let source_cone = M_bradford * source_illuminant_xyz;
  let target_cone = M_bradford * target_illuminant_xyz;

  var diag : mat3x3<f32> = mat3x3<f32>(target_cone.x / source_cone.x, 0.0, 0.0,
                                        0.0, target_cone.y / source_cone.y, 0.0,
                                        0.0, 0.0, target_cone.z / source_cone.z);

  let color_cone = M_bradford_inv * diag * M_bradford * xyz;

  return color_cone;
}

fn is_visible(p0 : vec3<f32>, p1 : vec3<f32>, exclude : u32) -> f32
{
  var ray : Ray;
  ray.origin = p0;
  ray.direction = normalize(p1 - p0);
  let shape_intersection : ShapeIntersection = intersect(ray, exclude);
  if(shape_intersection.hit)
  {
    if(length(shape_intersection.position - p1) < 0.0001)
    {
      return 1.0;
    }
  }
  return 0.0;
}

fn sample_light() -> vec3<f32>
{
  let light = planar_patches[2];
  let u = rand();
  let v = rand();
  let p = light.origin + u * light.edge1 + v * light.edge2;
  return p;
}

fn power_heuristic(nf : f32, f_pdf : f32, ng : f32, g_pdf : f32) -> f32
{
  let f = nf * f_pdf;
  let g = ng * g_pdf;
  return (f * f) / (f * f + g * g);
}

fn intersect(ray : Ray, exclude : u32) -> ShapeIntersection
{
  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;

  var t_max : f32 = INFINITY;
  for(var i : u32; i < arrayLength(&planar_patches); i++)
  {
    let temp_si = ray_patch_intersection_test(planar_patches[i], ray, 0.001, t_max, exclude);
    if(temp_si.hit)
    {
      shape_intersection = temp_si;
      t_max = temp_si.t;
    }
    
  }
  for(var i : u32; i < arrayLength(&spheres); i++)
  {
    let temp_si = ray_sphere_intersection(spheres[i], ray, 0.001, t_max, exclude);
    if(temp_si.hit)
    {
      shape_intersection = temp_si;
      t_max = temp_si.t;
    }
    
  }
  return shape_intersection;
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

fn ray_sphere_intersection(sphere : Sphere, ray : Ray, t_min : f32, t_max : f32, exclude : u32) -> ShapeIntersection
{
  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;

  if(exclude == sphere.index)
  {
    return shape_intersection;
  }

  let origin = ray.origin;
  let direction = ray.direction;
  let radius = sphere.radius;
  let center = sphere.center;
  let oc = origin - center;
  let a = dot(direction, direction);
  let b = 2.0 * dot(oc, direction);
  let c = dot(oc, oc) - radius * radius;
  let discriminant = b * b - 4.0 * a * c;

  if (discriminant < 0.0) {
    return shape_intersection;
  }

  var t : f32 = (-b - sqrt(discriminant)) / (2.0 * a);
  if (t < t_min || t > t_max) {
    t = (-b + sqrt(discriminant)) / (2.0 * a);
    if (t < t_min || t > t_max)
    {
      return shape_intersection;
    }
  }

  shape_intersection.position = ray_at(ray, t);
  shape_intersection.normal = normalize(shape_intersection.position - center);
  shape_intersection.t = t;
  shape_intersection.hit = true;
  shape_intersection.index = sphere.index;
  shape_intersection.albedo = sphere.albedo;
  shape_intersection.emission = sphere.emission;
  return shape_intersection;
}

fn ray_patch_intersection_test(planar_patch : PlanarPatch, ray : Ray, t_min : f32, t_max : f32, exclude : u32) -> ShapeIntersection
{

  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;

  if(exclude == planar_patch.index)
  {
    return shape_intersection;
  }


  let origin = ray.origin;
  let direction = ray.direction;
  let edge1 = planar_patch.edge1;
  let edge2 = planar_patch.edge2;
  var normal = normalize(cross((edge1), (edge2)));
  let origin_to_origin = planar_patch.origin - origin;
  var ndotd = dot(normal, direction);
  if(ndotd > 0)
  {
    normal = -normal;
  }
  ndotd = dot(normal, direction);
  if(abs(ndotd)  < 0.0001)
  {
    return shape_intersection;
  }

  let t = dot(normal, origin_to_origin) / (ndotd);
  if(t < t_min || t > t_max)
  {
    return shape_intersection;
  }
  let intersection = ray_at(ray, t);
  let m = intersection - planar_patch.origin;
  let u = dot(m, edge1) / dot(edge1, edge1);
  let v = dot(m, edge2) / dot(edge2, edge2);

  if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
  {
    return shape_intersection;
  }

  shape_intersection.hit = true;
  shape_intersection.position = intersection;
  shape_intersection.normal = normalize(normal);
  shape_intersection.t = t;
  shape_intersection.index = planar_patch.index;
  shape_intersection.albedo = planar_patch.albedo;
  shape_intersection.emission = planar_patch.emission;
  return shape_intersection;
}


/* Sampling */

fn uniformally_sample_hemisphere(normal: vec3<f32>) -> vec3<f32> {
  let u = rand();
  let v = rand();
  let theta = 2.0 * PI * u;
  let phi = acos(2.0 * v - 1.0);
  let x = cos(theta) * sin(phi);
  let y = sin(theta) * sin(phi);
  let z = cos(phi);
  var sample = vec3<f32>(x, y, z);
  if(dot(sample, normal) < 0.0)
  {
    sample = -sample;
  }
  return sample;
}

fn cosine_weighted_sample_hemisphere(normal : vec3<f32>) -> vec4<f32> {
  let u = rand();
  let v = rand();
  let r = sqrt(u);
  let theta = 2.0 * PI * v;
  let x = r * cos(theta);
  let y = r * sin(theta);
  let z = sqrt(max(0.0, 1.0 - u));

  var up : vec3<f32>;
  if(abs(normal.z) < 0.999)
  {
    up = vec3<f32>(0.0, 0.0, 1.0);
  }
  else
  {
    up = vec3<f32>(1.0, 0.0, 0.0);
  }
  let tangent = normalize(cross(up, normal));
  let bitangent = cross(normal, tangent);
  let direction = tangent * x + bitangent * y + normal * z;
  return vec4<f32>(direction, z/PI);
}


fn uniformally_sample_rectangle(rect : PlanarPatch) -> vec3<f32>
{
  let u = rand();
  let v = rand();
  let point = rect.origin + u * rect.edge1 + v * rect.edge2;
  return point;
}

fn sample_circle(radius : f32) -> vec2<f32>
{
    let r = sqrt(rand()) * radius;
    let theta = rand() * 2.0 * PI;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

fn sample_surface_circle(center : vec3<f32>, radius : f32) -> vec3<f32>
{
    let offset = sample_circle(radius);
    return vec3<f32>(center.x + offset.x, center.y , center.z + offset.y);
}


fn sample_circle_cosine_weighted(radius : f32) -> vec2<f32>
{
  let r = sqrt(rand()) * radius;
  let theta = rand() * 2.0 * PI;

  let u = rand();
  let v = rand();
  let phi = 2.0 * PI * u;
  let cosTheta = sqrt(1.0 - v);
  let sinTheta = sqrt(v);

  let x = cos(phi) * sinTheta;
  let y = sin(phi) * sinTheta;
  return vec2<f32>(r * x, r * y);
}


/* Utilties to generate random numbers */

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

fn pcg4d()
{
    seed = seed * 1664525u + 1013904223u;
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
    seed = seed ^ (seed >> vec4<u32>(16u, 16u, 16u, 16u));
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
}

fn rand() -> f32
{
    pcg4d(); 
    return f32(seed.x & 0x00ffffffu) / f32(0x01000000);
}

fn sample_CIE_X(lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(CIE_X[lambdas.x + 40], CIE_X[lambdas.y + 40], CIE_X[lambdas.z + 40], CIE_X[lambdas.w + 40]);
}

fn sample_CIE_Y(lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(CIE_Y[lambdas.x + 40], CIE_Y[lambdas.y + 40], CIE_Y[lambdas.z + 40], CIE_Y[lambdas.w + 40]);
}

fn sample_CIE_Z(lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(CIE_Z[lambdas.x + 40], CIE_Z[lambdas.y + 40], CIE_Z[lambdas.z + 40], CIE_Z[lambdas.w + 40]);
}

fn spectral_to_xyz_m() -> vec3<f32>
{


    var xyz = vec3<f32>(0.0, 0.0, 0.0);
    let integ = 106.856895;


    for(var i : u32 = 0; i < 1000; i++)
    {
        let sampled_spec : array<vec4<f32>, 2> = sample_light_spectrum();
        let lambdas = vec4<u32>(u32(sampled_spec[1].x), u32(sampled_spec[1].y), u32(sampled_spec[1].z), u32(sampled_spec[1].w));
        let spec = sampled_spec[0];
        let X_BAR = sample_CIE_X(lambdas);
        let Y_BAR = sample_CIE_Y(lambdas);
        let Z_BAR = sample_CIE_Z(lambdas);


        let radiance = spec * 300;

        xyz.x += (dot(X_BAR, radiance) /4);
        xyz.y += (dot(Y_BAR, radiance) /4);
        xyz.z += (dot(Z_BAR, radiance) /4);
    }
    xyz = xyz / integ;
    xyz = xyz / 1000.0;

    return xyz;
}

fn spectral_to_xyz(radianc : vec4<f32>, lambdas : vec4<u32>) -> vec3<f32>
{
    var xyz = vec3<f32>(0.0, 0.0, 0.0);
    let X_BAR = sample_CIE_X(lambdas);
    let Y_BAR = sample_CIE_Y(lambdas);
    let Z_BAR = sample_CIE_Z(lambdas);
    let integ = 106.856895;

    let radiance = radianc * 300;

    xyz.x = (dot(X_BAR, radiance) /4)/integ;
    xyz.y = (dot(Y_BAR, radiance) /4)/integ;
    xyz.z = (dot(Z_BAR, radiance) /4)/integ;

    return xyz;
}

fn xyz_to_rgb(rgb : vec3<f32>) -> vec3<f32>
{
    var r =  3.2404542 * rgb.x + -1.5371385 * rgb.y + -0.4985314 * rgb.z;
    var g = -0.9692660 * rgb.x +  1.8760108 * rgb.y +  0.0415560 * rgb.z;
    var b =  0.0556434 * rgb.x + -0.2040259 * rgb.y +  1.0572252 * rgb.z;

    var rr = vec3<f32>(r, g, b);
    rr = rr * 2.2;
    //rr = rr / (rr + vec3<f32>(1.0));

    if(r < 0.0031308)
    {
        rr.r = 12.92 * rr.r;
    }
    else
    {
        rr.r = 1.055 * pow(rr.r, 1.0/2.4) - 0.055;
    }

    if(g < 0.0031308)
    {
        rr.g = 12.92 * rr.g;
    }
    else
    {
        rr.g = 1.055 * pow(rr.g, 1.0/2.4) - 0.055;
    }

    if(b < 0.0031308)
    {
        rr.b = 12.92 * rr.b;
    }
    else
    {
        rr.b = 1.055 * pow(rr.b, 1.0/2.4) - 0.055;
    }

    return rr;


}

var<private> seed : vec4<u32> = vec4<u32>(0);


const CIE_X : array<f32, 471>  = array<f32, 471>(
    // CIE X function values
    0.0001299000,   0.0001458470,   0.0001638021,   0.0001840037,   0.0002066902,
    0.0002321000,   0.0002607280,   0.0002930750,   0.0003293880,   0.0003699140,
    0.0004149000,   0.0004641587,   0.0005189860,   0.0005818540,   0.0006552347,
    0.0007416000,   0.0008450296,   0.0009645268,   0.001094949,    0.001231154,
    0.001368000,    0.001502050,    0.001642328,    0.001802382,    0.001995757,
    0.002236000,    0.002535385,    0.002892603,    0.003300829,    0.003753236,
    0.004243000,    0.004762389,    0.005330048,    0.005978712,    0.006741117,
    0.007650000,    0.008751373,    0.01002888,     0.01142170,     0.01286901,
    0.01431000,     0.01570443,     0.01714744,     0.01878122,     0.02074801,
    0.02319000,     0.02620736,     0.02978248,     0.03388092,     0.03846824,
    0.04351000,     0.04899560,     0.05502260,     0.06171880,     0.06921200,
    0.07763000,     0.08695811,     0.09717672,     0.1084063,      0.1207672,
    0.1343800,      0.1493582,      0.1653957,      0.1819831,      0.1986110,
    0.2147700,      0.2301868,      0.2448797,      0.2587773,      0.2718079,
    0.2839000,      0.2949438,      0.3048965,      0.3137873,      0.3216454,
    0.3285000,      0.3343513,      0.3392101,      0.3431213,      0.3461296,
    0.3482800,      0.3495999,      0.3501474,      0.3500130,      0.3492870,
    0.3480600,      0.3463733,      0.3442624,      0.3418088,      0.3390941,
    0.3362000,      0.3331977,      0.3300411,      0.3266357,      0.3228868,
    0.3187000,      0.3140251,      0.3088840,      0.3032904,      0.2972579,
    0.2908000,      0.2839701,      0.2767214,      0.2689178,      0.2604227,
    0.2511000,      0.2408475,      0.2298512,      0.2184072,      0.2068115,
    0.1953600,      0.1842136,      0.1733273,      0.1626881,      0.1522833,
    0.1421000,      0.1321786,      0.1225696,      0.1132752,      0.1042979,
    0.09564000,     0.08729955,     0.07930804,     0.07171776,     0.06458099,
    0.05795001,     0.05186211,     0.04628152,     0.04115088,     0.03641283,
    0.03201000,     0.02791720,     0.02414440,     0.02068700,     0.01754040,
    0.01470000,     0.01216179,     0.009919960,    0.007967240,    0.006296346,
    0.004900000,    0.003777173,    0.002945320,    0.002424880,    0.002236293,
    0.002400000,    0.002925520,    0.003836560,    0.005174840,    0.006982080,
    0.009300000,    0.01214949,     0.01553588,     0.01947752,     0.02399277,
    0.02910000,     0.03481485,     0.04112016,     0.04798504,     0.05537861,
    0.06327000,     0.07163501,     0.08046224,     0.08973996,     0.09945645,
    0.1096000,      0.1201674,      0.1311145,      0.1423679,      0.1538542,
    0.1655000,      0.1772571,      0.1891400,      0.2011694,      0.2133658,
    0.2257499,      0.2383209,      0.2510668,      0.2639922,      0.2771017,
    0.2904000,      0.3038912,      0.3175726,      0.3314384,      0.3454828,
    0.3597000,      0.3740839,      0.3886396,      0.4033784,      0.4183115,
    0.4334499,      0.4487953,      0.4643360,      0.4800640,      0.4959713,
    0.5120501,      0.5282959,      0.5446916,      0.5612094,      0.5778215,
    0.5945000,      0.6112209,      0.6279758,      0.6447602,      0.6615697,
    0.6784000,      0.6952392,      0.7120586,      0.7288284,      0.7455188,
    0.7621000,      0.7785432,      0.7948256,      0.8109264,      0.8268248,
    0.8425000,      0.8579325,      0.8730816,      0.8878944,      0.9023181,
    0.9163000,      0.9297995,      0.9427984,      0.9552776,      0.9672179,
    0.9786000,      0.9893856,      0.9995488,      1.0090892,      1.0180064,
    1.0263000,      1.0339827,      1.0409860,      1.0471880,      1.0524667,
    1.0567000,      1.0597944,      1.0617992,      1.0628068,      1.0629096,
    1.0622000,      1.0607352,      1.0584436,      1.0552244,      1.0509768,
    1.0456000,      1.0390369,      1.0313608,      1.0226662,      1.0130477,
    1.0026000,      0.9913675,      0.9793314,      0.9664916,      0.9528479,
    0.9384000,      0.9231940,      0.9072440,      0.8905020,      0.8729200,
    0.8544499,      0.8350840,      0.8149460,      0.7941860,      0.7729540,
    0.7514000,      0.7295836,      0.7075888,      0.6856022,      0.6638104,
    0.6424000,      0.6215149,      0.6011138,      0.5811052,      0.5613977,
    0.5419000,      0.5225995,      0.5035464,      0.4847436,      0.4661939,
    0.4479000,      0.4298613,      0.4120980,      0.3946440,      0.3775333,
    0.3608000,      0.3444563,      0.3285168,      0.3130192,      0.2980011,
    0.2835000,      0.2695448,      0.2561184,      0.2431896,      0.2307272,
    0.2187000,      0.2070971,      0.1959232,      0.1851708,      0.1748323,
    0.1649000,      0.1553667,      0.1462300,      0.1374900,      0.1291467,
    0.1212000,      0.1136397,      0.1064650,      0.09969044,     0.09333061,
    0.08740000,     0.08190096,     0.07680428,     0.07207712,     0.06768664,
    0.06360000,     0.05980685,     0.05628216,     0.05297104,     0.04981861,
    0.04677000,     0.04378405,     0.04087536,     0.03807264,     0.03540461,
    0.03290000,     0.03056419,     0.02838056,     0.02634484,     0.02445275,
    0.02270000,     0.02108429,     0.01959988,     0.01823732,     0.01698717,
    0.01584000,     0.01479064,     0.01383132,     0.01294868,     0.01212920,
    0.01135916,     0.01062935,     0.009938846,    0.009288422,    0.008678854,
    0.008110916,    0.007582388,    0.007088746,    0.006627313,    0.006195408,
    0.005790346,    0.005409826,    0.005052583,    0.004717512,    0.004403507,
    0.004109457,    0.003833913,    0.003575748,    0.003334342,    0.003109075,
    0.002899327,    0.002704348,    0.002523020,    0.002354168,    0.002196616,
    0.002049190,    0.001910960,    0.001781438,    0.001660110,    0.001546459,
    0.001439971,    0.001340042,    0.001246275,    0.001158471,    0.001076430,
    0.0009999493,   0.0009287358,   0.0008624332,   0.0008007503,   0.0007433960,
    0.0006900786,   0.0006405156,   0.0005945021,   0.0005518646,   0.0005124290,
    0.0004760213,   0.0004424536,   0.0004115117,   0.0003829814,   0.0003566491,
    0.0003323011,   0.0003097586,   0.0002888871,   0.0002695394,   0.0002515682,
    0.0002348261,   0.0002191710,   0.0002045258,   0.0001908405,   0.0001780654,
    0.0001661505,   0.0001550236,   0.0001446219,   0.0001349098,   0.0001258520,
    0.0001174130,   0.0001095515,   0.0001022245,   0.00009539445,  0.00008902390,
    0.00008307527,  0.00007751269,  0.00007231304,  0.00006745778,  0.00006292844,
    0.00005870652,  0.00005477028,  0.00005109918,  0.00004767654,  0.00004448567,
    0.00004150994,  0.00003873324,  0.00003614203,  0.00003372352,  0.00003146487,
    0.00002935326,  0.00002737573,  0.00002552433,  0.00002379376,  0.00002217870,
    0.00002067383,  0.00001927226,  0.00001796640,  0.00001674991,  0.00001561648,
    0.00001455977,  0.00001357387,  0.00001265436,  0.00001179723,  0.00001099844,
    0.00001025398,  0.000009559646, 0.000008912044, 0.000008308358, 0.000007745769,
    0.000007221456, 0.000006732475, 0.000006276423, 0.000005851304, 0.000005455118,
    0.000005085868, 0.000004741466, 0.000004420236, 0.000004120783, 0.000003841716,
    0.000003581652, 0.000003339127, 0.000003112949, 0.000002902121, 0.000002705645,
    0.000002522525, 0.000002351726, 0.000002192415, 0.000002043902, 0.000001905497,
    0.000001776509, 0.000001656215, 0.000001544022, 0.000001439440, 0.000001341977,
    0.000001251141);

const CIE_Y : array<f32, 471>  = array<f32, 471>(0.000003917000,  0.000004393581,  0.000004929604,  0.000005532136,  0.000006208245,
    0.000006965000,  0.000007813219,  0.000008767336,  0.000009839844,  0.00001104323,
    0.00001239000,   0.00001388641,   0.00001555728,   0.00001744296,   0.00001958375,
    0.00002202000,   0.00002483965,   0.00002804126,   0.00003153104,   0.00003521521,
    0.00003900000,   0.00004282640,   0.00004691460,   0.00005158960,   0.00005717640,
    0.00006400000,   0.00007234421,   0.00008221224,   0.00009350816,   0.0001061361,
    0.0001200000,    0.0001349840,    0.0001514920,    0.0001702080,    0.0001918160,
    0.0002170000,    0.0002469067,    0.0002812400,    0.0003185200,    0.0003572667,
    0.0003960000,    0.0004337147,    0.0004730240,    0.0005178760,    0.0005722187,
    0.0006400000,    0.0007245600,    0.0008255000,    0.0009411600,    0.001069880,
    0.001210000,     0.001362091,     0.001530752,     0.001720368,     0.001935323,
    0.002180000,     0.002454800,     0.002764000,     0.003117800,     0.003526400,
    0.004000000,     0.004546240,     0.005159320,     0.005829280,     0.006546160,
    0.007300000,     0.008086507,     0.008908720,     0.009767680,     0.01066443,
    0.01160000,      0.01257317,      0.01358272,      0.01462968,      0.01571509,
    0.01684000,      0.01800736,      0.01921448,      0.02045392,      0.02171824,
    0.02300000,      0.02429461,      0.02561024,      0.02695857,      0.02835125,
    0.02980000,      0.03131083,      0.03288368,      0.03452112,      0.03622571,
    0.03800000,      0.03984667,      0.04176800,      0.04376600,      0.04584267,
    0.04800000,      0.05024368,      0.05257304,      0.05498056,      0.05745872,
    0.06000000,      0.06260197,      0.06527752,      0.06804208,      0.07091109,
    0.07390000,      0.07701600,      0.08026640,      0.08366680,      0.08723280,
    0.09098000,      0.09491755,      0.09904584,      0.1033674,       0.1078846,
    0.1126000,       0.1175320,       0.1226744,       0.1279928,       0.1334528,
    0.1390200,       0.1446764,       0.1504693,       0.1564619,       0.1627177,
    0.1693000,       0.1762431,       0.1835581,       0.1912735,       0.1994180,
    0.2080200,       0.2171199,       0.2267345,       0.2368571,       0.2474812,
    0.2586000,       0.2701849,       0.2822939,       0.2950505,       0.3085780,
    0.3230000,       0.3384021,       0.3546858,       0.3716986,       0.3892875,
    0.4073000,       0.4256299,       0.4443096,       0.4633944,       0.4829395,
    0.5030000,       0.5235693,       0.5445120,       0.5656900,       0.5869653,
    0.6082000,       0.6293456,       0.6503068,       0.6708752,       0.6908424,
    0.7100000,       0.7281852,       0.7454636,       0.7619694,       0.7778368,
    0.7932000,       0.8081104,       0.8224962,       0.8363068,       0.8494916,
    0.8620000,       0.8738108,       0.8849624,       0.8954936,       0.9054432,
    0.9148501,       0.9237348,       0.9320924,       0.9399226,       0.9472252,
    0.9540000,       0.9602561,       0.9660074,       0.9712606,       0.9760225,
    0.9803000,       0.9840924,       0.9874812,       0.9903128,       0.9928116,
    0.9949501,       0.9967108,       0.9980983,       0.9991120,       0.9997482,
    1.0000000,       0.9998567,       0.9993046,       0.9983255,       0.9968987,
    0.9950000,       0.9926005,       0.9897426,       0.9864444,       0.9827241,
    0.9786000,       0.9740837,       0.9691712,       0.9638568,       0.9581349,
    0.9520000,       0.9454504,       0.9384992,       0.9311628,       0.9234576,
    0.9154000,       0.9070064,       0.8982772,       0.8892048,       0.8797816,
    0.8700000,       0.8598613,       0.8493920,       0.8386220,       0.8275813,
    0.8163000,       0.8047947,       0.7930820,       0.7811920,       0.7691547,
    0.7570000,       0.7447541,       0.7324224,       0.7200036,       0.7074965,
    0.6949000,       0.6822192,       0.6694716,       0.6566744,       0.6438448,
    0.6310000,       0.6181555,       0.6053144,       0.5924756,       0.5796379,
    0.5668000,       0.5539611,       0.5411372,       0.5283528,       0.5156323,
    0.5030000,       0.4904688,       0.4780304,       0.4656776,       0.4534032,
    0.4412000,       0.4290800,       0.4170360,       0.4050320,       0.3930320,
    0.3810000,       0.3689184,       0.3568272,       0.3447768,       0.3328176,
    0.3210000,       0.3093381,       0.2978504,       0.2865936,       0.2756245,
    0.2650000,       0.2547632,       0.2448896,       0.2353344,       0.2260528,
    0.2170000,       0.2081616,       0.1995488,       0.1911552,       0.1829744,
    0.1750000,       0.1672235,       0.1596464,       0.1522776,       0.1451259,
    0.1382000,       0.1315003,       0.1250248,       0.1187792,       0.1127691,
    0.1070000,       0.1014762,       0.09618864,      0.09112296,      0.08626485,
    0.08160000,      0.07712064,      0.07282552,      0.06871008,      0.06476976,
    0.06100000,      0.05739621,      0.05395504,      0.05067376,      0.04754965,
    0.04458000,      0.04175872,      0.03908496,      0.03656384,      0.03420048,
    0.03200000,      0.02996261,      0.02807664,      0.02632936,      0.02470805,
    0.02320000,      0.02180077,      0.02050112,      0.01928108,      0.01812069,
    0.01700000,      0.01590379,      0.01483718,      0.01381068,      0.01283478,
    0.01192000,      0.01106831,      0.01027339,      0.009533311,     0.008846157,
    0.008210000,     0.007623781,     0.007085424,     0.006591476,     0.006138485,
    0.005723000,     0.005343059,     0.004995796,     0.004676404,     0.004380075,
    0.004102000,     0.003838453,     0.003589099,     0.003354219,     0.003134093,
    0.002929000,     0.002738139,     0.002559876,     0.002393244,     0.002237275,
    0.002091000,     0.001953587,     0.001824580,     0.001703580,     0.001590187,
    0.001484000,     0.001384496,     0.001291268,     0.001204092,     0.001122744,
    0.001047000,     0.0009765896,    0.0009111088,    0.0008501332,    0.0007932384,
    0.0007400000,    0.0006900827,    0.0006433100,    0.0005994960,    0.0005584547,
    0.0005200000,    0.0004839136,    0.0004500528,    0.0004183452,    0.0003887184,
    0.0003611000,    0.0003353835,    0.0003114404,    0.0002891656,    0.0002684539,
    0.0002492000,    0.0002313019,    0.0002146856,    0.0001992884,    0.0001850475,
    0.0001719000,    0.0001597781,    0.0001486044,    0.0001383016,    0.0001287925,
    0.0001200000,    0.0001118595,    0.0001043224,    0.00009733560,   0.00009084587,
    0.00008480000,   0.00007914667,   0.00007385800,   0.00006891600,   0.00006430267,
    0.00006000000,   0.00005598187,   0.00005222560,   0.00004871840,   0.00004544747,
    0.00004240000,   0.00003956104,   0.00003691512,   0.00003444868,   0.00003214816,
    0.00003000000,   0.00002799125,   0.00002611356,   0.00002436024,   0.00002272461,
    0.00002120000,   0.00001977855,   0.00001845285,   0.00001721687,   0.00001606459,
    0.00001499000,   0.00001398728,   0.00001305155,   0.00001217818,   0.00001136254,
    0.00001060000,   0.000009885877,  0.000009217304,  0.000008592362,  0.000008009133,
    0.000007465700,  0.000006959567,  0.000006487995,  0.000006048699,  0.000005639396,
    0.000005257800,  0.000004901771,  0.000004569720,  0.000004260194,  0.000003971739,
    0.000003702900,  0.000003452163,  0.000003218302,  0.000003000300,  0.000002797139,
    0.000002607800,  0.000002431220,  0.000002266531,  0.000002113013,  0.000001969943,
    0.000001836600,  0.000001712230,  0.000001596228,  0.000001488090,  0.000001387314,
    0.000001293400,  0.000001205820,  0.000001124143,  0.000001048009,  0.0000009770578,
    0.0000009109300, 0.0000008492513, 0.0000007917212, 0.0000007380904, 0.0000006881098,
    0.0000006415300, 0.0000005980895, 0.0000005575746, 0.0000005198080, 0.0000004846123,
    0.0000004518100
    );

const CIE_Z : array<f32, 471>  = array<f32, 471>(0.0006061000,   0.0006808792,   0.0007651456,   0.0008600124,   0.0009665928,
    0.001086000,    0.001220586,    0.001372729,    0.001543579,    0.001734286,
    0.001946000,    0.002177777,    0.002435809,    0.002731953,    0.003078064,
    0.003486000,    0.003975227,    0.004540880,    0.005158320,    0.005802907,
    0.006450001,    0.007083216,    0.007745488,    0.008501152,    0.009414544,
    0.01054999,     0.01196580,     0.01365587,     0.01558805,     0.01773015,
    0.02005001,     0.02251136,     0.02520288,     0.02827972,     0.03189704,
    0.03621000,     0.04143771,     0.04750372,     0.05411988,     0.06099803,
    0.06785001,     0.07448632,     0.08136156,     0.08915364,     0.09854048,
    0.1102000,      0.1246133,      0.1417017,      0.1613035,      0.1832568,
    0.2074000,      0.2336921,      0.2626114,      0.2947746,      0.3307985,
    0.3713000,      0.4162091,      0.4654642,      0.5196948,      0.5795303,
    0.6456000,      0.7184838,      0.7967133,      0.8778459,      0.9594390,
    1.0390501,      1.1153673,      1.1884971,      1.2581233,      1.3239296,
    1.3856000,      1.4426352,      1.4948035,      1.5421903,      1.5848807,
    1.6229600,      1.6564048,      1.6852959,      1.7098745,      1.7303821,
    1.7470600,      1.7600446,      1.7696233,      1.7762637,      1.7804334,
    1.7826000,      1.7829682,      1.7816998,      1.7791982,      1.7758671,
    1.7721100,      1.7682589,      1.7640390,      1.7589438,      1.7524663,
    1.7441000,      1.7335595,      1.7208581,      1.7059369,      1.6887372,
    1.6692000,      1.6475287,      1.6234127,      1.5960223,      1.5645280,
    1.5281000,      1.4861114,      1.4395215,      1.3898799,      1.3387362,
    1.2876400,      1.2374223,      1.1878243,      1.1387611,      1.0901480,
    1.0419000,      0.9941976,      0.9473473,      0.9014531,      0.8566193,
    0.8129501,      0.7705173,      0.7294448,      0.6899136,      0.6521049,
    0.6162000,      0.5823286,      0.5504162,      0.5203376,      0.4919673,
    0.4651800,      0.4399246,      0.4161836,      0.3938822,      0.3729459,
    0.3533000,      0.3348578,      0.3175521,      0.3013375,      0.2861686,
    0.2720000,      0.2588171,      0.2464838,      0.2347718,      0.2234533,
    0.2123000,      0.2011692,      0.1901196,      0.1792254,      0.1685608,
    0.1582000,      0.1481383,      0.1383758,      0.1289942,      0.1200751,
    0.1117000,      0.1039048,      0.09666748,     0.08998272,     0.08384531,
    0.07824999,     0.07320899,     0.06867816,     0.06456784,     0.06078835,
    0.05725001,     0.05390435,     0.05074664,     0.04775276,     0.04489859,
    0.04216000,     0.03950728,     0.03693564,     0.03445836,     0.03208872,
    0.02984000,     0.02771181,     0.02569444,     0.02378716,     0.02198925,
    0.02030000,     0.01871805,     0.01724036,     0.01586364,     0.01458461,
    0.01340000,     0.01230723,     0.01130188,     0.01037792,     0.009529306,
    0.008749999,    0.008035200,    0.007381600,    0.006785400,    0.006242800,
    0.005749999,    0.005303600,    0.004899800,    0.004534200,    0.004202400,
    0.003900000,    0.003623200,    0.003370600,    0.003141400,    0.002934800,
    0.002749999,    0.002585200,    0.002438600,    0.002309400,    0.002196800,
    0.002100000,    0.002017733,    0.001948200,    0.001889800,    0.001840933,
    0.001800000,    0.001766267,    0.001737800,    0.001711200,    0.001683067,
    0.001650001,    0.001610133,    0.001564400,    0.001513600,    0.001458533,
    0.001400000,    0.001336667,    0.001270000,    0.001205000,    0.001146667,
    0.001100000,    0.001068800,    0.001049400,    0.001035600,    0.001021200,
    0.001000000,    0.0009686400,   0.0009299200,   0.0008868800,   0.0008425600,
    0.0008000000,   0.0007609600,   0.0007236800,   0.0006859200,   0.0006454400,
    0.0006000000,   0.0005478667,   0.0004916000,   0.0004354000,   0.0003834667,
    0.0003400000,   0.0003072533,   0.0002831600,   0.0002654400,   0.0002518133,
    0.0002400000,   0.0002295467,   0.0002206400,   0.0002119600,   0.0002021867,
    0.0001900000,   0.0001742133,   0.0001556400,   0.0001359600,   0.0001168533,
    0.0001000000,   0.00008613333,  0.00007460000,  0.00006500000,  0.00005693333,
    0.00004999999,  0.00004416000,  0.00003948000,  0.00003572000,  0.00003264000,
    0.00003000000,  0.00002765333,  0.00002556000,  0.00002364000,  0.00002181333,
    0.00002000000,  0.00001813333,  0.00001620000,  0.00001420000,  0.00001213333,
    0.00001000000,  0.000007733333, 0.000005400000, 0.000003200000, 0.000001333333,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000);


const white_spectrum : array<f32, 76>  = array<f32, 76>(0.343, 0.445, 0.551, 0.624, 0.665, 0.687, 0.708, 0.723, 0.715, 0.710, 0.745, 0.758, 0.739, 0.767, 0.777, 0.765, 0.751, 0.745, 0.748, 0.729, 0.745, 0.757, 0.753, 0.750, 0.746, 0.747, 0.735, 0.732, 0.739, 0.734, 0.725, 0.721, 0.733, 0.725, 0.732, 0.743, 0.744, 0.748, 0.728, 0.716, 0.733, 0.726, 0.713, 0.740, 0.754, 0.764, 0.752, 0.736, 0.734, 0.741, 0.740, 0.732, 0.745, 0.755, 0.751, 0.744, 0.731, 0.733, 0.744, 0.731, 0.712, 0.708, 0.729, 0.730, 0.727, 0.707, 0.703, 0.729, 0.750, 0.760, 0.751, 0.739, 0.724, 0.730, 0.740, 0.737);
const green_spectrum : array<f32, 76> = array<f32, 76> (0.092, 0.096, 0.098, 0.097, 0.098, 0.095, 0.095, 0.097, 0.095, 0.094, 0.097, 0.098, 0.096, 0.101, 0.103, 0.104, 0.107, 0.109, 0.112, 0.115, 0.125, 0.140, 0.160, 0.187, 0.229, 0.285, 0.343, 0.390, 0.435, 0.464, 0.472, 0.476, 0.481, 0.462, 0.447, 0.441, 0.426, 0.406, 0.373, 0.347, 0.337, 0.314, 0.285, 0.277, 0.266, 0.250, 0.230, 0.207, 0.186, 0.171, 0.160, 0.148, 0.141, 0.136, 0.130, 0.126, 0.123, 0.121, 0.122, 0.119, 0.114, 0.115, 0.117, 0.117, 0.118, 0.120, 0.122, 0.128, 0.132, 0.139, 0.144, 0.146, 0.150, 0.152, 0.157, 0.159);
const red_spectrum : array<f32, 76> = array<f32, 76>   (0.040, 0.046, 0.048, 0.053, 0.049, 0.050, 0.053, 0.055, 0.057, 0.056, 0.059, 0.057, 0.061, 0.061, 0.060, 0.062, 0.062, 0.062, 0.061, 0.062, 0.060, 0.059, 0.057, 0.058, 0.058, 0.058, 0.056, 0.055, 0.056, 0.059, 0.057, 0.055, 0.059, 0.059, 0.058, 0.059, 0.061, 0.061, 0.063, 0.063, 0.067, 0.068, 0.072, 0.080, 0.090, 0.099, 0.124, 0.154, 0.192, 0.255, 0.287, 0.349, 0.402, 0.443, 0.487, 0.513, 0.558, 0.584, 0.620, 0.606, 0.609, 0.651, 0.612, 0.610, 0.650, 0.638, 0.627, 0.620, 0.630, 0.628, 0.642, 0.639, 0.657, 0.639, 0.635, 0.642);

fn adjust_spec() -> array<f32, 76>
{
  let min_factor = 0.6;
  let max_factor = 1.8;
  let mid_index : f32 = 38;
  var adjustec_spec = array<f32, 76>(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
  for(var i : u32 = 0; i < 76; i++)
  {
    let factor = min_factor + (max_factor - min_factor) * (1.0 - abs(f32(i) - mid_index) / mid_index);
    adjustec_spec[i] = factor * green_spectrum[i];
  }
  return adjustec_spec;
}

fn sample_light_spectrum() -> array<vec4<f32>,2>
{
  let u = rand();
  let lambda : f32 = mix(lambda_min, lambda_max, u); // from 400 to 700
  var i :  u32 = u32((lambda - lambda_min)); // from 0 to 299
  var lambdas : vec4<f32> = vec4<f32>(0);
  lambdas[0] = f32(i);
  let val : f32 = sample_piecewise_linear_light(i);
  i = (i + 4) % 300;
  lambdas[1] = f32(i);
  let val2 : f32 = sample_piecewise_linear_light(i);
  i = (i + 4) % 300;
  lambdas[2] = f32(i);
  let val3 : f32 = sample_piecewise_linear_light(i);
  i = (i + 4) % 300;
  lambdas[3] = f32(i);
  let val4 : f32 = sample_piecewise_linear_light(i);
  return array<vec4<f32>,2>(vec4<f32>(val, val2, val3, val4), lambdas);
}

fn sample_piecewise_linear_white(lambdas: vec4<u32>) -> vec4<f32>
{
  var val1 : f32 = 0.0;
  //lambdas[x] is from 0 to 299
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[0] < i * 4u)
    {
      val1 = mix(white_spectrum[i - 1u], white_spectrum[i], f32(lambdas[0] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val2: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[1] < i * 4u)
    {
      val2 = mix(white_spectrum[i - 1u], white_spectrum[i], f32(lambdas[1] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }
  var val3: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[2] < i * 4u)
    {
      val3 = mix(white_spectrum[i - 1u], white_spectrum[i], f32(lambdas[2] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }
  var val4: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[3] < i * 4u)
    {
      val4 = mix(white_spectrum[i - 1u], white_spectrum[i], f32(lambdas[3] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }
  return vec4<f32>(val1, val2, val3, val4);
}

fn sample_piecewise_linear_red(lambdas : vec4<u32>) -> vec4<f32>
{
  var val1 : f32 = 0.0;
  //lambdas[x] is from 0 to 299
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[0] < i * 4u)
    {
      val1 = mix(red_spectrum[i - 1u], red_spectrum[i], f32(lambdas[0] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val2: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[1] < i * 4u)
    {
      val2 = mix(red_spectrum[i - 1u], red_spectrum[i], f32(lambdas[1] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val3: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[2] < i * 4u)
    {
      val3 = mix(red_spectrum[i - 1u], red_spectrum[i], f32(lambdas[2] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val4: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[3] < i * 4u)
    {
      val4 = mix(red_spectrum[i - 1u], red_spectrum[i], f32(lambdas[3] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  return vec4<f32>(val1, val2, val3, val4);
}

fn sample_piecewise_linear_green(lambdas : vec4<u32>) -> vec4<f32>
{
  var val1 : f32 = 0.0;
  //lambdas[x] is from 0 to 299
  //let green_spectrum = adjust_spec();

  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[0] < i * 4u)
    {
      val1 = mix(green_spectrum[i - 1u], green_spectrum[i], f32(lambdas[0] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val2: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[1] < i * 4u)
    {
      val2 = mix(green_spectrum[i - 1u], green_spectrum[i], f32(lambdas[1] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val3: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[2] < i * 4u)
    {
      val3 = mix(green_spectrum[i - 1u], green_spectrum[i], f32(lambdas[2] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  var val4: f32 = 0.0;
  for(var i : u32 = 0u; i < 76u; i = i + 1u)
  {
    if(lambdas[3] < i * 4u)
    {
      val4 = mix(green_spectrum[i - 1u], green_spectrum[i], f32(lambdas[3] - (i - 1u) * 4u) / 4.0);
      break;
    }
  }

  return vec4<f32>(val1, val2, val3, val4);
}

fn sample_piecewise_linear_light(lambda_index : u32) -> f32
{
  if(lambda_index < 100)
  {
    return mix(light_spectrum[0], light_spectrum[1], f32(lambda_index) / 100.0);
  }
  else if(lambda_index < 200)
  {
    return mix(light_spectrum[1], light_spectrum[2], f32(lambda_index - 100) / 100.0);
  }
  else
  {
    return mix(light_spectrum[2], light_spectrum[3], f32(lambda_index - 200) / 100.0);
  }
}