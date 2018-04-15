extern crate image;
extern crate rand;
extern crate csv;
extern crate minifb;

use std::fs::File;
use std::path::Path;
use minifb::{MouseButton, MouseMode, Window, Key, Scale, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const GRID_W: u32 = 6;
const GRID_H: u32 = 3;

fn load_csv() -> Vec<csv::StringRecord>{
    let f = File::open(&Path::new("/Users/goncalopalaio/Dropbox/2017/kaggle-facial-keypoints-detection/python/test_results.csv")).expect("Error reading csv");
    let mut reader = csv::Reader::from_reader(f);
    let mut vec: Vec<csv::StringRecord> = Vec::new();

    for r in reader.records() {
        //println!("{:?}", r);

        match r {
            Ok(record) => vec.push(record),
            _ => (),
        }
    }

    vec
}

fn paste_image(buffer: &mut Vec<u32>, image_index: u32, grid_w_pos: u32, grid_h_pos: u32, point_list: &Vec<csv::StringRecord>) {
    let image = image::open(format!("/Users/goncalopalaio/Dropbox/2017/kaggle-facial-keypoints-detection/images/test/{}_img.png", image_index)).unwrap().to_rgba();
    //image = image::imageops::blur(&image, 5.1);
    let points = point_list.get(image_index as usize).unwrap();
    //println!("current points: {:?}", points);
    let offset_x = grid_w_pos * 96;
    let offset_y = grid_h_pos * 96;
    
    for i in 0..96 {
        for j in 0..96 {            
            //buffer[i + j * WIDTH] = 0x00ffffff
            let index = ((i + offset_x) + (j + offset_y) * WIDTH as u32) as usize;
            let pixel = image[(i as u32,j as u32)][0] as u32;
            buffer[index] = (pixel << 16) + (pixel << 8) + (pixel);
        }
    }

    let mut r = 0;
    while r < 30 - 1 {
        
        let x = points.get(r).unwrap().parse::<usize>().unwrap() + (offset_x as usize);
        let y = points.get(r + 1).unwrap().parse::<usize>().unwrap() + (offset_y as usize);

        buffer[x + y * WIDTH] = 0x00ff0000;
        //buffer[1 + x + y * WIDTH] = 0x00ff0000;
        //buffer[2 + x + y * WIDTH] = 0x00ff0000;
        r += 2;
    }

}

fn main() {
    println!("Hello, world!");

    let mut points = load_csv();
    //let image = image::open(image_path).unwrap().to_rgba();
    //let image_dims = image.dimensions();
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = match Window::new("Gallery - Press ESC to exit", WIDTH, HEIGHT,
                                       WindowOptions {
                                           scale: Scale::X2,
                                           ..WindowOptions::default()
                                       }) {
        Ok(win) => win,
        Err(err) => {
            println!("Unable to create window {}", err);
            return;
        }
    };
    let mut image_start_index = 0;
    while window.is_open() && !window.is_key_down(Key::Escape) {

        let mut img_idx = image_start_index;
        for gw in 0..GRID_W {
            for gh in 0..GRID_H {
                paste_image(&mut buffer, img_idx, gw, gh, &points);        
                img_idx += 1;
            }
        }
    
        window.get_keys().map(|keys| {
            for t in keys {
                match t {
                    Key::W => println!("holding w!"),
                    Key::T => println!("holding t!"),
                    Key::R => {
                        println!("holding r! reloading csv");
                        points = load_csv();   
                    },
                    Key::Q => {
                        println!("holding q!");
                        image_start_index += 1;
                        println!("{:?}", image_start_index);
                    },
                    _ => (),
                }
            }
        });

        window.get_mouse_pos(MouseMode::Discard).map(|mouse| {
            let screen_pos = ((mouse.1 as usize) * WIDTH) + mouse.0 as usize;
            
            if window.get_mouse_down(MouseButton::Left) {
                println!("{:?}", window.get_unscaled_mouse_pos(MouseMode::Discard).unwrap());
                buffer[screen_pos] = 0x00ffffff;
            }

            if window.get_mouse_down(MouseButton::Right) {
                buffer[screen_pos] = 0;
            }
        });

        window.get_scroll_wheel().map(|scroll| {
            println!("Scrolling {} - {}", scroll.0, scroll.1);
            image_start_index += 1;
        });

        // We unwrap here as we want this code to exit if it fails
        window.update_with_buffer(&buffer).unwrap();
    }
    
    println!("Exiting");
}

