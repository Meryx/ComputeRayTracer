@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> alt_color_buffer : array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> sample : u32;
@group(0) @binding(3) var<storage> planar_patches : array<PlanarPatch>;
@group(0) @binding(4) var<storage> camera : Camera;
@group(0) @binding(5) var<storage> CIE : array<array<f32,471>,3>;
@group(0) @binding(6) var<storage> spectra : array<array<f32, 301>>;
@group(0) @binding(7) var<storage> lights : array<Primitive>;
@group(0) @binding(8) var<storage> primitives : array<Primitive>;

const PI: f32 = 3.14159265359;
const INFINITY : f32 = 0x7F800000;
const MAX_U32_VALUE : u32 = 0xFFFFFFFF;
const MAXDEPTH : u32 = 100;
const GRID_SIZE : u32 = 16;
const lambda_min : f32 = 400.0;
const lambda_max : f32 = 700.0;
const DIFFUSE : u32 = 0;
const LIGHT : u32 = 1;
const GLASS : u32 = 2;

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
  emission_index: u32,
  reflectance_index: u32,
  material: u32,
  index: u32
};

struct Primitive {
  category: u32,
  data1: vec3<f32>,
  data2: vec3<f32>,
  data3: vec3<f32>,
  data4: vec4<u32>,
}

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>
};

struct ShapeIntersection {
  position: vec3<f32>,
  normal: vec3<f32>,
  emission_index: u32,
  reflectance_index: u32,
  material: u32
};

struct IntersectionContext {
  t_min: f32,
  t_max: f32,
  index: u32,
  exclude: u32,
  hit: bool,
  ray_origin: vec3<f32>,
  ray_direction: vec3<f32>
}

struct Intersection {
  shape_intersection: ShapeIntersection,
  context: IntersectionContext
}

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
  // if(sample > 3)
  // {
  //   return;
  // }

    let screen_size: vec2<u32> = vec2<u32>(u32(camera.viewport_width), u32(camera.viewport_height));
    let screen_pos : vec2<u32> = vec2<u32>(u32(GlobalInvocationID.x), u32(GlobalInvocationID.y));
    screen_pos_g = screen_pos;
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }

    // let s = primitives[arrayLength(&primitives) - 1];
    // let val = f32(s.category);
    // let c = vec3<f32>(val, val, val);
    // textureStore(framebuffer, screen_pos, vec4<f32>(c/1000000000, 1.0));
    // return;
    
    seed = vec4<u32>(screen_pos.y, screen_pos.x * 100, sample, tea(screen_pos.x, screen_pos.y * 100));

    let ray : Ray = camera_ray(screen_pos, screen_size);
    let wavelengths = sample_wavelengths();

    let radiance : vec4<f32> = path_trace(ray, wavelengths);
    // return;
    let xyz_color = spectral_to_xyz(radiance, wavelengths);
  
    let pixel_index = screen_pos.x + screen_pos.y * screen_size.x;
    alt_color_buffer[pixel_index] += xyz_color;
    let accumulated_xyz_color = alt_color_buffer[pixel_index];
    let averaged_xyz_color = accumulated_xyz_color / f32(sample);
    let rgb_color = xyz_to_linear_rgb(averaged_xyz_color);
    var rgb_color_ldr = tone_map(rgb_color, 2.2);
    gamma_correct(&rgb_color_ldr);

    textureStore(framebuffer, screen_pos, vec4<f32>(rgb_color_ldr, 1.0));
    
}

fn path_trace(input_ray : Ray, wavelengths : vec4<u32>) -> vec4<f32>
{
  var ray : Ray = input_ray;
  var depth : u32 = 0;
  var accumulated_radiance : vec4<f32> = vec4<f32>(0.0);
  var beta : vec4<f32> = vec4<f32>(1.0);
  var last_bounce_pdf : f32 = 1.0;
  var exclude : u32 = MAX_U32_VALUE;
  var BRDF = vec4<f32>(1.0);
  var specular_bounce : bool = false;
  var prev_intersection : Intersection;
  var etaScale = 1.0;
  var inTransmission = false;

  while(true)
  {
    let intersection : Intersection = intersect(ray, exclude, false);
    let shape_intersection : ShapeIntersection = intersection.shape_intersection;
    let intersection_context : IntersectionContext = intersection.context;


    
    if(!intersection_context.hit)
    {
      break;
    }

    exclude = intersection_context.index;

    var material = shape_intersection.material;
    if(material == LIGHT)
    {

      let le = sample_spectrum(shape_intersection.emission_index, wavelengths);
      if(depth == 0 || specular_bounce)
      {
        accumulated_radiance += beta * le;
      } 
      else 
      {
        let pdf_l = compute_light_pdf(intersection);
        let weight_b = power_heuristic(1, last_bounce_pdf, 1, pdf_l);
        accumulated_radiance += weight_b * le * beta;
      }
      break;
    }


    if(depth >= MAXDEPTH)
    {
      break;
    }


    if(inTransmission)
    {
      let distance = distance(intersection_context.ray_origin, shape_intersection.position);
      let extinction = sample_spectrum(arrayLength(&spectra) - 1, wavelengths);
      let attenuation = vec4<f32>(exp(-extinction.x * distance), exp(-extinction.y * distance), exp(-extinction.z * distance), exp(-extinction.w * distance));
      beta *= attenuation;
    }


    if(shape_intersection.material == DIFFUSE)
    {
      BRDF = sample_spectrum(shape_intersection.reflectance_index, wavelengths) / PI;

      let le = compute_light_radiance(intersection, wavelengths);
      accumulated_radiance += BRDF * le * beta;
      
      let new_direction = cosine_weighted_sample_hemisphere(shape_intersection.normal, &last_bounce_pdf);
      let cos_theta = abs(dot(shape_intersection.normal, new_direction));

      beta *= BRDF * cos_theta / last_bounce_pdf;
      ray.origin = shape_intersection.position;
      ray.direction = new_direction;
      specular_bounce = false;
              // textureStore(framebuffer, screen_pos_g, vec4<f32>(0.5,0,0, 1.0));
        // return vec4<f32>(0.0);
      prev_intersection = intersection;
      // if(inTransmission)
      // {
      //   beta /= beta;
      // }
      // return vec4<f32>(0.0);
    }
                  // return accumulated_radiance;


    if(shape_intersection.material == GLASS)
    {
      let eta1 = 1.0;
      let eta2 = 1.5;
      var eta = eta1 / eta2;
      let cos_theta = dot(shape_intersection.normal, ray.direction);
      let reflected = fresnel_s(ray.direction, shape_intersection.normal, eta1, eta2);
      let transmitted = 1 - reflected;
      var pr = reflected;
      let pt = 1 - reflected;
      let u = rand();
      var new_direction = vec3<f32>(0.0);
      var current_direction = ray.direction;
      var current_normal = shape_intersection.normal;
      // current_normal = -current_normal;


      // pr = fresnel(cos_theta, 1.125);
      // textureStore(framebuffer, screen_pos_g, vec4<f32>(pr , pr, pr, 1.0));
      // return vec4<f32>(0.0);

      if(cos_theta > 0)
      {
        eta = 1.0 / eta;
        // cos_theta = -cos_theta;
        current_normal = -current_normal;
      }
      new_direction = (reflect(current_direction, current_normal));
              let test_direction = (refract(current_direction, current_normal, eta));

      if(u < pr / (pr + pt))
      {
        new_direction = (reflect(current_direction, current_normal));
        // textureStore(framebuffer, screen_pos_g, vec4<f32>((current_normal + 1) / 2, 1.0));
        // return vec4<f32>(0.0);

        ray.origin = shape_intersection.position;
        specular_bounce = true;
              exclude = MAX_U32_VALUE;

      }
      else
      {
                specular_bounce = true;

              exclude = MAX_U32_VALUE;

        new_direction = (refract(current_direction, current_normal, eta));

        
          new_direction = normalize(new_direction);
          ray.origin = shape_intersection.position;
          // specular_bounce = true;
          beta *= (eta * eta);
          etaScale /= (eta * eta);
          inTransmission = !inTransmission;
        
      }





        
      
      ray.direction = new_direction;
      
       
    }


    let rbeta = beta * etaScale;
    let max_beta_component = max(rbeta.x, max(rbeta.y, rbeta.z));
    if(depth > 1 && max_beta_component < 1)
    {
      let q = max(0, 1 - max_beta_component);
      if(rand() < q)
      {
        break;
      }
      beta /= 1 - q;
    }


    depth++;
  }
  return accumulated_radiance;
}

fn power_heuristic(nf : f32, f_pdf : f32, ng : f32, g_pdf : f32) -> f32
{
  let f = nf * f_pdf;
  let g = ng * g_pdf;
  return (f * f) / (f * f + g * g);
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

//======================== SPECTRA FUNCTIONS ========================
fn sample_spectrum(index : u32,  lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(spectra[index][lambdas.x], spectra[index][lambdas.y], spectra[index][lambdas.z], spectra[index][lambdas.w]);
}

fn sample_wavelengths() -> vec4<u32>
{
  let u = rand();
  let range : u32 = u32(lambda_max - lambda_min) + 1;
  let lambda : u32 = u32(mix(0, lambda_max - lambda_min + 1, u));

  return vec4<u32>(lambda, (lambda + 4) % range, (lambda + 8) % range, (lambda + 12) % range);
}

fn sample_CIE_X(lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(CIE[0][lambdas.x + 40], CIE[0][lambdas.y + 40], CIE[0][lambdas.z + 40], CIE[0][lambdas.w + 40]);
}

fn sample_CIE_Y(lambdas : vec4<u32>) -> vec4<f32>
{
  return vec4<f32>(CIE[1][lambdas.x + 40], CIE[1][lambdas.y + 40], CIE[1][lambdas.z + 40], CIE[1][lambdas.w + 40]);
}

fn sample_CIE_Z(lambdas : vec4<u32>) -> vec4<f32>
{
  
  return vec4<f32>(CIE[2][lambdas.x + 40], CIE[2][lambdas.y + 40], CIE[2][lambdas.z + 40], CIE[2][lambdas.w + 40]);
}

//======================== LIGHTS FUNCTIONS ========================
fn sample_lights() -> Primitive
{
  let u = rand();
  let range : f32 = f32(arrayLength(&lights));
  let index : u32 = u32(mix(0, range, u));
  return lights[index];
}

fn sample_light(light : Primitive) -> vec3<f32>
{
  let u = rand();
  let v = rand();
  let p = light.data1 + u * light.data2 + v * light.data3;
  return p;
}

fn compute_light_pdf(intersection : Intersection) -> f32
{
  let shape_intersection : ShapeIntersection = intersection.shape_intersection;
  let intersection_context : IntersectionContext = intersection.context;
  
  let light : Primitive = lights[shape_intersection.emission_index];
  let light_area : f32 = length(light.data2) * length(light.data3);
  let light_area_pdf : f32 = 1.0 / light_area;

  let abs_cos_theta : f32 = max(0.00001, abs(dot(shape_intersection.normal, -intersection_context.ray_direction)));
  let distance : f32 = length(shape_intersection.position - intersection_context.ray_origin);
  let distance_squared : f32 = pow(distance, 2);
  let geometric_term : f32 = abs_cos_theta / distance_squared;
  let light_solid_angle_pdf : f32 = light_area_pdf / geometric_term;

  let number_of_lights : f32 = f32(arrayLength(&lights));
  let light_selection_pdf : f32 = 1.0 / number_of_lights;
  
  let light_pdf : f32 = light_selection_pdf * light_solid_angle_pdf;
  return light_pdf;
}

fn compute_light_radiance(intersection : Intersection, wavelengths : vec4<u32>) -> vec4<f32>
{
  let shape_intersection : ShapeIntersection = intersection.shape_intersection;
  let context : IntersectionContext = intersection.context;
  let light = sample_lights();
  let point_on_light = sample_light(light);
  let light_dir = normalize(point_on_light - shape_intersection.position);
  let shadow_intersection = shadow_intersect(Ray(shape_intersection.position, light_dir), light.data4.w, context.index);

  let cos_theta = max(0, dot(shape_intersection.normal, light_dir));
  let spec = sample_spectrum(light.data4.x, wavelengths);

  let le = spec * cos_theta;

  let pdf_l = compute_light_pdf(shadow_intersection);

  if(shadow_intersection.context.hit)
  {
    let pdf_b = cos_theta / PI;
    let weight_l = power_heuristic(1, pdf_l, 1, pdf_b);
    
    let t = le * weight_l / pdf_l;
    // if(length(t) > 0.1)
    // {
    //   return vec4<f32>(0);
    // }
    return t;
  }
  return vec4<f32>(0.0);
}

//======================== COLOR FUNCTIONS ========================
fn sample_CIE(wavelengths : vec4<u32>) -> mat4x3<f32>
{
  let X_BAR = sample_CIE_X(wavelengths);
  let Y_BAR = sample_CIE_Y(wavelengths);
  let Z_BAR = sample_CIE_Z(wavelengths);
  return transpose(mat3x4<f32>(X_BAR, Y_BAR, Z_BAR));
}

fn spectral_to_xyz(radiance : vec4<f32>, wavelengths : vec4<u32>) -> vec3<f32>
{
    let CIE_transform = sample_CIE(wavelengths);
    let integ = 106.856895;
    let lambda_range = lambda_max - lambda_min;
    var xyz = CIE_transform * radiance;
    return xyz * lambda_range / (integ * 4);
}

fn xyz_to_linear_rgb(rgb : vec3<f32>) -> vec3<f32>
{
    var r =  3.2404542 * rgb.x + -1.5371385 * rgb.y + -0.4985314 * rgb.z;
    var g = -0.9692660 * rgb.x +  1.8760108 * rgb.y +  0.0415560 * rgb.z;
    var b =  0.0556434 * rgb.x + -0.2040259 * rgb.y +  1.0572252 * rgb.z;
    return vec3<f32>(r, g, b);
}

fn tone_map(rgb : vec3<f32>, exposure : f32) -> vec3<f32>
{
    return vec3(1.0) - exp(-rgb * exposure);
}

fn gamma_correct(rgb : ptr<function, vec3<f32>>)
{
  if((*rgb).r < 0.0031308)
    {
        (*rgb).r *= 12.92;
    }
    else
    {
        (*rgb).r = 1.055 * pow((*rgb).r, 1.0/2.4) - 0.055;
    }
    if((*rgb).g < 0.0031308)
    {
        (*rgb).g *= 12.92 * (*rgb).g;
    }
    else
    {
        (*rgb).g = 1.055 * pow((*rgb).g, 1.0/2.4) - 0.055;
    }
    if((*rgb).b < 0.0031308)
    {
        (*rgb).b = 12.92 * (*rgb).b;
    }
    else
    {
        (*rgb).b = 1.055 * pow((*rgb).b, 1.0/2.4) - 0.055;
    }
}

//======================== CAMERA FUNCTIONS ========================
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

//======================== INTERSECTION FUNCTIONS ========================
fn intersect(ray : Ray, exclude : u32, shadow : bool) -> Intersection
{
  var shape_intersection : ShapeIntersection;
  var intersection_context : IntersectionContext = create_intersection_context(exclude);

  for(var i : u32; i < arrayLength(&primitives) ; i++)
  {
    if(shadow && primitives[i].data4.z == GLASS)
    {
      continue;
    }
    ray_intersection(primitives[i], ray, &intersection_context, &shape_intersection); 
  }

  return Intersection(shape_intersection, intersection_context);
}

fn ray_intersection(primitive : Primitive, ray : Ray, 
                    context: ptr<function, IntersectionContext>, 
                    shape_intersection : ptr<function, ShapeIntersection>)
{
  let category : u32 = primitive.category;
  if(category == 0)
  {
    let exclude = (*context).exclude;
    let index = primitive.data4.w;
    if(exclude == index)
    {
      return;
    }

    let edge1 = primitive.data2;
    let edge2 = primitive.data3;
    var normal = normalize(cross((edge1), (edge2)));

    let direction = ray.direction;
    var ndotd = dot(normal, direction);

    if(ndotd > 0)
    {
      normal = -normal;
    }
    ndotd = dot(normal, direction);
    if(abs(ndotd)  < 0.0001)
    {
      return;
    }

    let origin = ray.origin;
    let origin_to_origin = primitive.data1 - origin;

    let t = dot(normal, origin_to_origin) / (ndotd);
    let t_min = (*context).t_min;
    let t_max = (*context).t_max;
    if(t < t_min || t > t_max)
    {
      return;
    }
    let intersection = ray_at(ray, t);
    let m = intersection - primitive.data1;
    let u = dot(m, edge1) / dot(edge1, edge1);
    let v = dot(m, edge2) / dot(edge2, edge2);

    if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
    {
      return;
    }

    (*shape_intersection).position = intersection;
    (*shape_intersection).normal = normal;
    (*shape_intersection).emission_index = primitive.data4.x;
    (*shape_intersection).reflectance_index = primitive.data4.y;
    (*shape_intersection).material = primitive.data4.z;

    (*context).t_max = t;
    (*context).index = index;
    (*context).hit = true;
    (*context).ray_origin = origin;
    (*context).ray_direction = direction;
    return;
  }
  if(category == 1)
  {
    let exclude = (*context).exclude;
    let index = primitive.data4.w;
    if(exclude == index)
    {
      return;
    }
    let center = primitive.data1;
    let radius = primitive.data2.x;
    let radius_squared = radius * radius;
    let origin = ray.origin;
    let direction = ray.direction;
    let center_to_origin = origin - center;
    let a = dot(direction, direction);
    let b = 2.0 * dot(direction, center_to_origin);
    let c = dot(center_to_origin, center_to_origin) - radius_squared;
    let discriminant = b * b - 4.0 * a * c;
    if(discriminant <= 0.0)
    {
      return;
    }
    var t = (-b - sqrt(discriminant)) / (2.0 * a);
    let t_min = (*context).t_min;
    let t_max = (*context).t_max;
    if(t < t_min || t > t_max)
    {
      t = (-b + sqrt(discriminant)) / (2.0 * a);
      if(t < t_min || t > t_max)
      {
        return;
      }
    }
    let intersection = ray_at(ray, t);
    let normal = normalize(intersection - center);
    
    (*shape_intersection).position = intersection;
    (*shape_intersection).normal = normal;
    (*shape_intersection).emission_index = primitive.data4.x;
    (*shape_intersection).reflectance_index = primitive.data4.y;
    (*shape_intersection).material = primitive.data4.z;

    (*context).t_max = t;
    (*context).index = primitive.data4.w;
    (*context).hit = true;
    (*context).ray_origin = origin;
    (*context).ray_direction = direction;
  }
}

fn ray_patch_intersection(planar_patch : PlanarPatch, ray : Ray, 
                          context: ptr<function, IntersectionContext>, 
                          shape_intersection : ptr<function, ShapeIntersection>)
{

  let exclude = (*context).exclude;
  let index = planar_patch.index;
  if(exclude == index)
  {
    return;
  }


  let edge1 = planar_patch.edge1;
  let edge2 = planar_patch.edge2;
  var normal = normalize(cross((edge1), (edge2)));

  let direction = ray.direction;
  var ndotd = dot(normal, direction);

  if(ndotd > 0)
  {
    normal = -normal;
  }
  ndotd = dot(normal, direction);
  if(abs(ndotd)  < 0.0001)
  {
    return;
  }

  let origin = ray.origin;
  let origin_to_origin = planar_patch.origin - origin;

  let t = dot(normal, origin_to_origin) / (ndotd);
  let t_min = (*context).t_min;
  let t_max = (*context).t_max;
  if(t < t_min || t > t_max)
  {
    return;
  }
  let intersection = ray_at(ray, t);
  let m = intersection - planar_patch.origin;
  let u = dot(m, edge1) / dot(edge1, edge1);
  let v = dot(m, edge2) / dot(edge2, edge2);

  if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
  {
    return;
  }

  (*shape_intersection).position = intersection;
  (*shape_intersection).normal = normal;
  (*shape_intersection).emission_index = planar_patch.emission_index;
  (*shape_intersection).reflectance_index = planar_patch.reflectance_index;
  (*shape_intersection).material = planar_patch.material;

  (*context).t_max = t;
  (*context).index = index;
  (*context).hit = true;
  (*context).ray_origin = origin;
  (*context).ray_direction = direction;
}

fn shadow_intersect(ray : Ray, include : u32, exclude : u32) -> Intersection
{
  var intersection : Intersection = intersect(ray, exclude, false);
  if(intersection.context.index != include)
  {
    intersection.context.hit = false;
  }
  return intersection;
}



fn create_intersection_context(exclude : u32) -> IntersectionContext
{
  /* PARAMETERS:
     t_min         : f32
     t_max         : f32
     index         : u32
     exclude       : u32 
     hit           : bool
     ray_origin    : vec3<f32>
     ray_direction : vec3<f32>
  */
  return IntersectionContext(0.001, INFINITY, MAX_U32_VALUE, exclude, false, vec3<f32>(0), vec3<f32>(0));
}

//======================== SHAPE FUNCTIONS ========================
fn compute_patch_normal(planar_patch : PlanarPatch, direction : vec3<f32>) -> vec3<f32>
{
    var normal = normalize(cross((planar_patch.edge1), (planar_patch.edge2)));
    let ndotd = dot(normal, direction);
    if(ndotd > 0)
    {
      normal = -normal;
    }
    return normal;
}

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

fn cosine_weighted_sample_hemisphere(normal : vec3<f32>, pdf : ptr<function, f32>) -> vec3<f32> {
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
  (*pdf) = z / PI;
  return vec3<f32>(direction);
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

//======================== MATERIAL FUNCTIONS ========================
fn fresnel_s(ray_dir: vec3<f32>, normal: vec3<f32>, eta1 : f32, eta2 : f32) -> f32 {
    var cosi = clamp(dot(ray_dir, normal), -1.0, 1.0);
    var eta = eta1 / eta2;

    if (cosi > 0.0) {
        eta = eta2 / eta1;
    }

    let sint2 = eta * eta * (1.0 - cosi * cosi);
    
    if (sint2 > 1.0) {
        return 1.0;  // Total internal reflection, return 1.0
    }

    var cost = sqrt(1.0 - sint2);

    cosi = abs(cosi);
    let Rs = ((eta1 * cosi) - (eta2 * cost)) / ((eta1 * cosi) + (eta2 * cost));
    let Rp = ((eta2 * cosi) - (eta1 * cost)) / ((eta2 * cosi) + (eta1 * cost));

    let reflectance = (Rs * Rs + Rp * Rp) / 2.0;

    return reflectance;
}
fn fresnel(cos_theta_input : f32, etat_input : f32) -> f32
{
  var cos_theta = clamp(cos_theta_input, -1, 1);
  var etat = etat_input;

  if(cos_theta < 0)
  {
    cos_theta = -cos_theta;
    etat = 1.0 / etat;
  }

  let sin2theta_i = 1.0 - cos_theta * cos_theta;
  let sin2theta_t = sin2theta_i / (etat * etat);
  if(sin2theta_t >= 1.0)
  {
    return 1.0;
  }

  let cos_theta_t = sqrt(1.0 - sin2theta_t);

  let r_parallel = (etat * cos_theta - cos_theta_t) / (etat * cos_theta + cos_theta_t);
  let r_perpendicular = (cos_theta - etat * cos_theta_t) / (cos_theta + etat * cos_theta_t);
  return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2.0;

}

//======================== RANDOM UTILITIES FUNCTIONS ========================
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

var<private> seed : vec4<u32> = vec4<u32>(0);
var<private> screen_pos_g : vec2<u32> = vec2<u32>(0);