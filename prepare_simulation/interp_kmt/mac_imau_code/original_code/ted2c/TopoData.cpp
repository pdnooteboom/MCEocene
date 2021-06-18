#include <iostream>
#include <fstream>
#include <cassert>
#include <limits>
using namespace std;

#include "TopoData.h"


int colormap[256][3] = { 
  {0,  0,   0}, {0, 4, 255}, {0, 7, 255}, {0, 11, 255}, 
  {0, 16, 255}, {0, 19, 255}, {0, 24, 255}, {0, 28, 255}, 
  {0, 31, 255}, {0, 36, 255}, {0, 39, 255}, {0, 43, 255}, 
  {0, 48, 255}, {0, 51, 255}, {0, 56, 255}, {0, 59, 255}, 
  {0, 63, 255}, {0, 68, 255}, {0, 71, 255}, {0, 76, 255}, 
  {0, 80, 255}, {0, 83, 255}, {0, 88, 255}, {0, 91, 255}, 
  {0, 95, 255}, {0, 100, 255}, {0, 103, 255}, {0, 108, 255}, 
  {0, 112, 255}, {0, 115, 255}, {0, 120, 255}, {0, 123, 255}, 
  {0, 127, 255}, {0, 132, 255}, {0, 135, 255}, {0, 140, 255}, 
  {0, 144, 255}, {0, 147, 255}, {0, 152, 255}, {0, 155, 255}, 
  {0, 159, 255}, {0, 164, 255}, {0, 167, 255}, {0, 172, 255}, 
  {0, 175, 255}, {0, 179, 255}, {0, 184, 255}, {0, 187, 255}, 
  {0, 192, 255}, {0, 196, 255}, {0, 199, 255}, {0, 204, 255}, 
  {0, 207, 255}, {0, 211, 255}, {0, 216, 255}, {0, 219, 255}, 
  {0, 224, 255}, {0, 228, 255}, {0, 231, 255}, {0, 236, 255}, 
  {0, 239, 255}, {0, 243, 255}, {0, 248, 255}, {0, 251, 255}, 
  {0, 255, 253}, {0, 255, 249}, {0, 255, 246}, {0, 255, 241}, 
  {0, 255, 238}, {0, 255, 234}, {0, 255, 229}, {0, 255, 226}, 
  {0, 255, 221}, {0, 255, 218}, {0, 255, 214}, {0, 255, 209}, 
  {0, 255, 206}, {0, 255, 201}, {0, 255, 197}, {0, 255, 194}, 
  {0, 255, 189}, {0, 255, 186}, {0, 255, 182}, {0, 255, 177}, 
  {0, 255, 174}, {0, 255, 169}, {0, 255, 165}, {0, 255, 162}, 
  {0, 255, 157}, {0, 255, 154}, {0, 255, 150}, {0, 255, 145}, 
  {0, 255, 142}, {0, 255, 137}, {0, 255, 133}, {0, 255, 130}, 
  {0, 255, 125}, {0, 255, 122}, {0, 255, 118}, {0, 255, 113}, 
  {0, 255, 110}, {0, 255, 105}, {0, 255, 102}, {0, 255, 98}, 
  {0, 255, 93}, {0, 255, 90}, {0, 255, 85}, {0, 255, 81}, 
  {0, 255, 78}, {0, 255, 73}, {0, 255, 70}, {0, 255, 66}, 
  {0, 255, 61}, {0, 255, 58}, {0, 255, 53}, {0, 255, 49}, 
  {0, 255, 46}, {0, 255, 41}, {0, 255, 38}, {0, 255, 34}, 
  {0, 255, 29}, {0, 255, 26}, {0, 255, 21}, {0, 255, 17}, 
  {0, 255, 14}, {0, 255, 9}, {0, 255, 6}, {0, 255, 2}, 
  {2, 255, 0}, {5, 255, 0}, {10, 255, 0}, {14, 255, 0}, 
  {17, 255, 0}, {22, 255, 0}, {25, 255, 0}, {30, 255, 0}, 
  {34, 255, 0}, {37, 255, 0}, {42, 255, 0}, {45, 255, 0}, 
  {49, 255, 0}, {54, 255, 0}, {57, 255, 0}, {62, 255, 0}, 
  {66, 255, 0}, {69, 255, 0}, {74, 255, 0}, {77, 255, 0}, 
  {81, 255, 0}, {86, 255, 0}, {89, 255, 0}, {94, 255, 0}, 
  {98, 255, 0}, {101, 255, 0}, {106, 255, 0}, {109, 255, 0}, 
  {113, 255, 0}, {118, 255, 0}, {121, 255, 0}, {126, 255, 0}, 
  {130, 255, 0}, {133, 255, 0}, {138, 255, 0}, {141, 255, 0}, 
  {146, 255, 0}, {150, 255, 0}, {153, 255, 0}, {158, 255, 0}, 
  {161, 255, 0}, {165, 255, 0}, {170, 255, 0}, {173, 255, 0}, 
  {178, 255, 0}, {182, 255, 0}, {185, 255, 0}, {190, 255, 0}, 
  {193, 255, 0}, {197, 255, 0}, {202, 255, 0}, {205, 255, 0}, 
  {210, 255, 0}, {214, 255, 0}, {217, 255, 0}, {222, 255, 0}, 
  {225, 255, 0}, {229, 255, 0}, {234, 255, 0}, {237, 255, 0}, 
  {242, 255, 0}, {246, 255, 0}, {249, 255, 0}, {254, 255, 0}, 
  {255, 252, 0}, {255, 247, 0}, {255, 243, 0}, {255, 240, 0}, 
  {255, 235, 0}, {255, 232, 0}, {255, 228, 0}, {255, 223, 0}, 
  {255, 220, 0}, {255, 215, 0}, {255, 211, 0}, {255, 208, 0}, 
  {255, 203, 0}, {255, 200, 0}, {255, 196, 0}, {255, 191, 0}, 
  {255, 188, 0}, {255, 183, 0}, {255, 179, 0}, {255, 176, 0}, 
  {255, 171, 0}, {255, 168, 0}, {255, 164, 0}, {255, 159, 0}, 
  {255, 156, 0}, {255, 151, 0}, {255, 147, 0}, {255, 144, 0}, 
  {255, 139, 0}, {255, 136, 0}, {255, 132, 0}, {255, 127, 0}, 
  {255, 124, 0}, {255, 119, 0}, {255, 116, 0}, {255, 112, 0}, 
  {255, 107, 0}, {255, 104, 0}, {255, 99, 0}, {255, 95, 0}, 
  {255, 92, 0}, {255, 87, 0}, {255, 84, 0}, {255, 80, 0}, 
  {255, 75, 0}, {255, 72, 0}, {255, 67, 0}, {255, 63, 0}, 
  {255, 60, 0}, {255, 55, 0}, {255, 52, 0}, {255, 48, 0}, 
  {255, 43, 0}, {255, 40, 0}, {255, 35, 0}, {255, 31, 0}, 
  {255, 28, 0}, {255, 23, 0}, {255, 20, 0}, {255, 16, 0}, 
  {255, 11, 0}, {255, 8, 0}, {255, 3, 0}, {255, 0, 0} 
};

// ----------------
// --- TopoData ---
// ----------------
//
//
//
TopoData::TopoData(void)
{
  valid = false;
  ds_width = ds_height = 0;
  ds_width = ds_height = 0;
  image = 0;
  elev  = 0;
  min   =  INT_MAX;
  max   = -INT_MAX;  
  mapVPWidth  = MAPWINDOW_WIDTH;
  mapVPHeight = MAPWINDOW_HEIGHT;
  zoomVPWidth  = (unsigned int) (MAPWINDOW_WIDTH * 0.8);
  zoomVPHeight = (unsigned int) (MAPWINDOW_HEIGHT * 0.8);
}


// ----------------
// --- TopoData ---
// ----------------
//
//
//
TopoData::TopoData(unsigned int w, unsigned int h, const char *filename)
{
  ds_width  = w;
  ds_height = h;
  min    =  INT_MAX;
  max    = -INT_MAX;
  aspectRatio = (float)w/(float)h;
  
  elev   = new int[ds_width * ds_height];
  assert(elev);
  
  image  = new unsigned char[ds_width * ds_height * 4];
  assert(image);

  // Check to see if we should read a data file.
  if (filename != 0) {
    if (readKMT(filename) < 0)
      valid = false;
  } else {
    memset((void *)elev, 0, sizeof(int) * ds_width * ds_height);
    memset((void *)image, 0, sizeof(unsigned char) * ds_width * ds_height * 4);
    min = max = 0;
    valid = true; // Assume an empty dataset is valid.
  }
}


// -----------------
// --- ~TopoData ---
// -----------------
//
//
//
TopoData::~TopoData(void)
{
  delete []elev;
  delete []image;
}


// -----------------
// --- elevation ---
// -----------------
//
//
//
int TopoData::elevation(int x, int y)
{
  if (x < 0 || x > (ds_width-1)) {
    cerr << "Warning!\n";
    cerr << "\tx coordinate value out of bounds in elevation() request.\n";
    cerr << "\tx = " << x << endl;
    return -1;
  }
  
  if (y < 0 || y > (ds_height-1)) {
    cerr << "Warning!\n";
    cerr << "\ty coordinate value out of bounds in elevation() request.\n";
    cerr << "\ty = " << y << endl;
    return -1;
  }

  return elev[y * ds_width + x];
}


void TopoData::setValue(int x, int y, int value)
{
  if (x < 0 || x > (ds_width-1)) {
    cerr << "Warning!\n";
    cerr << "\tx coordinate value out of bounds in setValue() request.\n";
    cerr << "\tx = " << x << endl;
    return;
  }
  
  if (y < 0 || y > (ds_height-1)) {
    cerr << "Warning!\n";
    cerr << "\ty coordinate value out of bounds in setValue() request.\n";
    cerr << "\ty = " << y << endl;
    return;
  }

  elev[y * ds_width + x] = value;
  buildImage();
}


// -------------
// --- pixel ---
// -------------
//
//
//
unsigned char TopoData::pixel(int x, int y)
{
  if (x < 0 || x > (ds_width-1)) {
    cerr << "Warning!\n";
    cerr << "\tx coordinate value out of bounds in pixel() request.\n";
    cerr << "\tx = " << x << endl;
    return 0;
  }
  
  if (y < 0 || y > (ds_height-1)) {
    cerr << "Warning!\n";
    cerr << "\ty coordinate value out of bounds in pixel() request.\n";
    cerr << "\ty = " << y << endl;
    return 0;
  }

  return image[y * (ds_width*4) + x];
}


// ------------------
// --- buildImage ---
// ------------------
//
//
//
void TopoData::buildImage(void)
{
  assert(image);
  assert(elev);
  
  cerr << "Constructing image from kmt field...\n";  

  // Set the min and max values.
  for(int i = 0; i < ds_width * ds_height; i++) {
    if (elev[i] < min)
      min = elev[i];
    if (elev[i] > max)
      max = elev[i];
  }

  cerr << "\tmin kmt value = " << min << endl;
  cerr << "\tmax kmt value = " << max << endl;  

  // Build the image.  KMT is backasswards in terms of what we want to
  // present to the user so we do an inverse here.
  int j;
  for(int  i = 0, j= 0; i < ds_width * ds_height; i++, j+=4) {
    int index = (unsigned char)((1.0 - (float)(max - elev[i]) /
                             (float)(max - min)) * 255.0);
    image[j]   = colormap[index][0];
    image[j+1] = colormap[index][1];    
    image[j+2] = colormap[index][2];    
    image[j+3] = 0;
  }
}


// ---------------
// --- readKMT ---
// ---------------
//
//
//
int TopoData::readKMT(const char *filename)
{
  assert(filename);
  ifstream kmtin(filename);
  if (!kmtin) {
    cerr << "Could not open kmt file for reading: " << filename << endl;
    valid = false;
    return -1;
  }

  cerr << "Reading kmt field data from: " << filename << endl;

  kmtin.read((char *)elev, ds_width * ds_height * sizeof(int));
  kmtin.close();
  
  // Our last step is to build an image of the kmt data.
  buildImage();

  return 0;
}


// ----------------
// --- writeKMT ---
// ----------------
//
//
//
int TopoData::writeKMT(const char *filename)
{
  assert(filename);
  ofstream kmtout(filename);
  if (!kmtout) {
    cerr << "Could not open kmt file for writing: " << filename << endl;
    return -1;
  }
  
  kmtout.write((const char *)elev, ds_width * ds_height * sizeof(int));
  kmtout.close();
  return 0;
}

// ------------------
// --- writeImage ---
// ------------------
//
//
//
int TopoData::writeImage(const char *filename)
{
  assert(filename);
  //writeRGB(filename, image, ds_width, ds_height);
  return 0;
}


  
