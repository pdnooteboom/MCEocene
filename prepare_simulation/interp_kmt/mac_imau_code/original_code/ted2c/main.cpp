#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
using namespace std;

#include "TopoData.h"
#include "Namelist.h"

string 	kmt_file, nml_input;

TopoData *tData = 0;

int       mapMenuID, zoomMapMenuID;
float     xZoomFactor = 1.0;
float     yZoomFactor = 1.0;
float     xZoomF      = 1.0;
float     yZoomF      = 1.0;
char      buf[256];
int       bufXPos = 0, bufYPos = 0;

int       zoomBoxSize = 25;

int mapWin, zoomMapWin;

typedef struct {
  int leftdown, middledown, rightdown;
} MouseState;

MouseState zoomMouseState;
MouseState mapMouseState;

void zoomWindowDisplay(void);

typedef struct {
  int x0, y0, x1, y1;
} ZoomBox;

ZoomBox zoomBox;
static bool zooming = false;
//GLvoid *pixes;
unsigned char *pixels;
GLsizei  displayDataWidth = 50, displayDataHeight = 50;


// ---------------
// --- saveKMT ---
// ---------------
//
//
//
void saveKMT(int id)
{
  if (id == 0) {
    char filename[512];
    cerr << "Enter the new KMT filename: ";
    cin >> filename;
    if (tData) {
      tData->writeKMT(filename);
      cerr << "\tSaved.\n";
    }
    return;
  }

  int x = zoomBox.x0 + zoomBoxSize;
  int y = zoomBox.y0 + zoomBoxSize;
  zoomBoxSize       = id;  
  zoomBox.x0        = x - zoomBoxSize; 
  zoomBox.x1        = x + zoomBoxSize; 
  zoomBox.y0        = y - zoomBoxSize; 
  zoomBox.y1        = y + zoomBoxSize;  
  displayDataWidth  = zoomBoxSize * 2;
  displayDataHeight = zoomBoxSize * 2;
  glutSetWindow(mapWin);  
  glutPostRedisplay();
  glutSetWindow(zoomMapWin);
  glutPostRedisplay();  
}


// ------------------------
// --- mapWindowReshape ---
// ------------------------
//
//
//
void mapWindowReshape(int w, int h)
{
  float aspectRatio;
  if(tData) { // Have we loaded a data set?
    aspectRatio = tData->getAspectRatio();
    if(aspectRatio > (float)w/(float)h) {
      // Set the viewport size in TopoData.
      tData->setMapVPWidth((unsigned int) w);
      h = (int)((float)w/aspectRatio);
      tData->setMapVPHeight((unsigned int)h);
      // This causes multiple redraws!!
      glutReshapeWindow(w, h); 
    } else {
      w = (int) ((float)h * aspectRatio);
      tData->setMapVPWidth((unsigned int) w);
      tData->setMapVPHeight((unsigned int) h);
      // This causes multiple redraws!!
      glutReshapeWindow(w, h);
    }
  } 

  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLfloat)w, 0.0, (GLfloat)h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // If there's data, zoom it to fit the resize.
  if (tData) {
    // Get the zoom factor.
    // viewport size divided by the data set size.
    xZoomFactor = (float)tData->getMapVPWidth() / (float)tData->getDSWidth();
    yZoomFactor = (float)tData->getMapVPHeight()/(float)tData->getDSHeight();
    // Clamp to limits on the SGI IR hardware...
    if (xZoomFactor > 128)
      xZoomFactor = 128;
    if (yZoomFactor > 128)
      yZoomFactor = 128;
    glPixelZoom(xZoomFactor, yZoomFactor);
  } // end if tData
}


// -------------------------
// --- zoomWindowReshape ---
// -------------------------
//
//
//
void zoomWindowReshape(int w, int h)
{
  float aspectRatio;

  if(tData) {
    aspectRatio = tData->getAspectRatio();
    if(aspectRatio > (float)w/(float)h) {
      // Set the viewport size in TopoData.
      tData->setZoomVPWidth((unsigned int) w);
      h = (int)((float)w/aspectRatio);

      // This causes multiple redraws!!
      glutReshapeWindow(w, h); 
    } else {
      w = (int) ((float)h * aspectRatio);
      tData->setZoomVPWidth((unsigned int) w);
      tData->setZoomVPHeight((unsigned int) h);
      // This causes multiple redraws!!
      glutReshapeWindow(w, h);
    }
  }

  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLfloat)w, 0.0, (GLfloat)h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // If there's data, zoom it to fit the resize.
  if (tData) {
    // Get the zoom factor.
    // viewport size divided by the data set size.
    xZoomF = (tData->getZoomVPWidth() / ((zoomBoxSize*2)/xZoomFactor));
    yZoomF = (tData->getZoomVPHeight()/ ((zoomBoxSize*2)/yZoomFactor));
    
    // Clamp to limits on the SGI IR hardware...
    if (xZoomF > 128)
      xZoomF = 128;
    if (yZoomF > 128)
      yZoomF = 128;
    glPixelZoom(xZoomF, yZoomF);
  }   // end if tData
}


void init(void)
{
  buf[0] = '\0';
  // Fast pixel drawing state.
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_DITHER);
  glDisable(GL_FOG);
  glDisable(GL_LIGHTING);
  glDisable(GL_LOGIC_OP);
  glDisable(GL_STENCIL_TEST);
  glDisable(GL_TEXTURE_1D);
  glDisable(GL_TEXTURE_2D);
  glPixelTransferi(GL_MAP_COLOR, GL_FALSE);
  glPixelTransferi(GL_RED_SCALE, 1);
  glPixelTransferi(GL_RED_BIAS, 0);
  glPixelTransferi(GL_GREEN_SCALE, 1);
  glPixelTransferi(GL_GREEN_BIAS, 0);
  glPixelTransferi(GL_BLUE_SCALE, 1);
  glPixelTransferi(GL_BLUE_BIAS, 0);
  glPixelTransferi(GL_ALPHA_SCALE, 1);
  glPixelTransferi(GL_ALPHA_BIAS, 0);

/*  glDisable(GL_CONVOLUTION_1D_EXT);
  glDisable(GL_CONVOLUTION_2D_EXT);  
  glDisable(GL_SEPARABLE_2D_EXT);  
  glDisable(GL_HISTOGRAM_EXT);
  glDisable(GL_MINMAX_EXT);
  glDisable(GL_TEXTURE_3D_EXT);
  glDisable(GL_MULTISAMPLE_SGIS);
*/  
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
}

void glutDisplayString(char *string)
{
  int i;
  for(i = 0; i < strlen(string); i++) {
    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, (int)string[i]);
  }
}

void mapWindowDisplay(void)
{
  glClear(GL_COLOR_BUFFER_BIT);
  if (tData) {
    glRasterPos2i(0, 0);
    glDrawPixels(tData->getDSWidth(),
                 tData->getDSHeight(),
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 tData->pixels());

    if (buf[0] != '\0') {
      glPushMatrix();
      glRasterPos2i(bufXPos, bufYPos);
      glColor3f(1.0, 1.0, 1.0);
      glutDisplayString(buf);
      glPopMatrix();
    }
  }

  if(zooming) {
    glColor3f(1.0, 1.0, 1.0);
    glLineWidth(4.0);    
    glBegin(GL_LINE_LOOP);
      glVertex2i(zoomBox.x0-1, zoomBox.y0-1);
      glVertex2i(zoomBox.x1+1, zoomBox.y0-1);
      glVertex2i(zoomBox.x1+1, zoomBox.y1+1);
      glVertex2i(zoomBox.x0-1, zoomBox.y1+1);
    glEnd();
  }
  
  glutSwapBuffers();  
}

void zoomWindowDisplay(void)
{
  //cerr << "ZOOM MAP map display" << endl;
  glClear(GL_COLOR_BUFFER_BIT);

  if(zooming) {

    int xskip = zoomBox.x0/xZoomFactor;
    int yskip = zoomBox.y0/yZoomFactor;
    glRasterPos2i(0, 0);
    xZoomF = (tData->getZoomVPWidth() / ((zoomBoxSize*2)/xZoomFactor));
    yZoomF = (tData->getZoomVPHeight()/ ((zoomBoxSize*2)/yZoomFactor));
    glPixelZoom(xZoomF, yZoomF);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, xskip);
    glPixelStorei(GL_UNPACK_ROW_LENGTH,  tData->getDSWidth());    
    glPixelStorei(GL_UNPACK_SKIP_ROWS,   yskip);
    glDrawPixels(int((zoomBoxSize*2.0)/xZoomFactor)+1,
                 int((zoomBoxSize*2.0)/yZoomFactor)+1,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 tData->pixels());

    if(buf[0] != '\0') {
      glPushMatrix();
      glRasterPos2i(bufXPos, bufYPos);
      glColor3f(1.0, 1.0, 1.0);
      glutDisplayString(buf);
      glPopMatrix();
    }
  }
  glutSwapBuffers();
}


void keyboard(unsigned char key, int x, int y)
{
  switch (key) {
    case 27: // ESC
      exit(0);
      break;
  }
}


void mapWindowMouse(int button, int state, int x, int y)
{
  int h = glutGet(GLUT_WINDOW_HEIGHT);
  int w = glutGet(GLUT_WINDOW_WIDTH);  
  y = h - y;

  switch(button) {
    case GLUT_RIGHT_BUTTON:
      mapMouseState.rightdown = (state==GLUT_DOWN);
      break;
    case GLUT_MIDDLE_BUTTON:
      mapMouseState.middledown = (state==GLUT_DOWN);
      break;
    case GLUT_LEFT_BUTTON:
      mapMouseState.leftdown = (state==GLUT_DOWN);
      zooming = true;

      if(mapMouseState.leftdown && zooming) {
        // make a box

        if ((x - zoomBoxSize) < 0)
          x += fabs(x - zoomBoxSize);
        if ((x + zoomBoxSize) > w)
          x = w - zoomBoxSize;

        if ((y - zoomBoxSize) < 0)
          y += fabs(y - zoomBoxSize);
        if ((y + zoomBoxSize) > h)
          y = h - zoomBoxSize;        
        #include <string>
        zoomBox.x0 = x - zoomBoxSize; 
        zoomBox.x1 = x + zoomBoxSize; 
        zoomBox.y0 = y - zoomBoxSize; 
        zoomBox.y1 = y + zoomBoxSize;
      }
      displayDataWidth = zoomBox.x1 - zoomBox.x0;
      displayDataHeight = zoomBox.y1 - zoomBox.y0;
      glutPostRedisplay();
      glutSetWindow(zoomMapWin);
      zoomWindowDisplay();
      break;
  }
}


void zoomWindowMotion( int x, int y)
{
  if (buf[0] != '\0') {
    int h = glutGet(GLUT_WINDOW_HEIGHT);
    int w = glutGet(GLUT_WINDOW_WIDTH);
    y = h - y;
    
    if (x < 0)
      x = 0;
    if (y < 0)
      y = 0;
    if (x >= w)
      x = w-1;
    if (y >= h)
      y = h - 1;

    int xloc = int(zoomBox.x0/xZoomFactor) + int(x / xZoomF);
    int yloc = int(zoomBox.y0/yZoomFactor) + int(y / yZoomF);
    int value = tData->elevation(xloc, yloc);
    
    sprintf(buf, "(%d, %d) = %d", xloc+1, yloc+1, value);
    bufYPos = 10;        
    bufXPos = 10;          
    glutPostRedisplay();    
  }
}

void zoomWindowMouse(int button, int state, int x, int y)
{
  int h = glutGet(GLUT_WINDOW_HEIGHT);
  int w = glutGet(GLUT_WINDOW_WIDTH);  
  y = h - y;

  if (x < 0)
    x = 0;
  if (y < 0)
    y = 0;
  if (x >= w)
    x = w-1;
  if (y >= h)
    y = h - 1;

  int xloc =  int(zoomBox.x0/xZoomFactor) + int(x / xZoomF);
  int yloc =  int(zoomBox.y0/yZoomFactor) + int(y / yZoomF);
  int value = tData->elevation(xloc, yloc);  
    
  switch(button) {
    case GLUT_RIGHT_BUTTON:
      zoomMouseState.rightdown = (state==GLUT_DOWN);
      break;
    case GLUT_MIDDLE_BUTTON:
      zoomMouseState.middledown = (state==GLUT_DOWN);
      if(zoomMouseState.middledown) {
        if (tData) {
          // Covert (x, y) back into grid location based on
          // zoom factors.
          sprintf(buf, "(%d, %d) = %d", xloc+1, yloc+1, value);
          bufYPos = 10;        
          bufXPos = 10;
        }
      } else {
        cerr << "Enter the new data value for ("
             << xloc+1 << ", " << yloc+1 << "): ";
        int newval;
        cin >> newval;
        cerr << "New value = " << newval << endl;
        buf[0] = '\0';
        if (tData) {
          tData->setValue(xloc, yloc, newval);

          // Force redraw on the main map window to
          // reflect changes.
          glutSetWindow(mapWin);

          glPixelZoom(xZoomFactor, yZoomFactor);
          mapWindowDisplay();
          glFlush();

          glutSetWindow(zoomMapWin);
          glPixelZoom(xZoomF, yZoomF);
          glFlush();
        }
      }
      break;
    case GLUT_LEFT_BUTTON:
      zoomMouseState.leftdown = (state==GLUT_DOWN);
      if(zoomMouseState.leftdown) {
        if (tData) {
          // Covert (x, y) back into grid location based on
          // zoom factors.
          sprintf(buf, "(%d, %d) = %d", xloc+1, yloc+1, value);
          bufYPos = 10;        
          bufXPos = 10;
        }
      } else {
        buf[0] = '\0';
      } 
      break;
		
  }
  glutPostRedisplay();
}


void mapWindowMotion(int x, int y)
{
  glutPostRedisplay();
}

void
enter_leave(int state)
{
  //printf("enter/leave %d = %s\n",
  //       glutGetWindow(),
  //       state == GLUT_LEFT ? "left" : "entered");
}

int main(int argc, char *argv[])
{
/*  CmdArgInt    xdim('x', "X-Dimension", "scalar",
                    "Dimension of data in X dimension.");
  CmdArgInt    ydim('y', "Y-Dimension", "scalar",
                    "Dimension of data in Y dimension.");  
  CmdArgStr    kmtFile("kmt-file", "KMT data file.");

  CmdLine      cmd(*argv, &xdim, &ydim, &kmtFile, NULL);
  CmdArgvIter  argl(--argc, ++argv);
  cmd.description("KMT field editor.");
  cmd.parse(argl);
*/
   // Get namelist file name from command line; default: "nml_input.txt".
    
    string nml_input("nml_input.txt");  // Works for "ted in" (argc == 2)
    if ((argc == 2) || (argc == 4))     // and "ted in > out" (argc == 4)
        nml_input.assign(argv[1]);      // but not "ted > out" (argc == 3)

	int		xdim = 0;		// value of zero will generate an error
	int		ydim = 0;
	string	kmt_file = string("NONE");
	
	Namelist control("control");
	control.entry("xdim", &xdim);
	control.entry("ydim", &ydim);
	control.entry("kmt_file", &kmt_file);
	control.entry("zoomBoxSize", &zoomBoxSize);
	
	control.print("initial values");
	control.read(nml_input);
	control.print("final values");
	
  int w = MAPWINDOW_WIDTH;
  int h = MAPWINDOW_HEIGHT;
  float aspectRatio;

  if (xdim == 0 || ydim == 0) {
    cerr << "You must specify the x and y dimensions of the kmt file.\n";
    return 1;
  }

  tData = new TopoData(xdim, ydim, kmt_file.c_str());
  assert(tData);
  //tData->writeImage("kmt.rgb");  // Sanity check...


  // Initialize the window to the same aspect ratio as the data set
  aspectRatio = (float)xdim/(float)ydim;
  if(aspectRatio > (float)w/(float)h) {
    h = int(w/aspectRatio);
  } else {
    w = int(h*aspectRatio);
  }

  glutInitWindowSize(w, h);
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  mapMenuID = glutCreateMenu(saveKMT);
  glutAddMenuEntry("4x4", 2);
  glutAddMenuEntry("8x8", 4);    
  glutAddMenuEntry("10x10", 5);
  glutAddMenuEntry("20x20", 10);
  glutAddMenuEntry("30x30", 15);
  glutAddMenuEntry("40x40", 20);
  glutAddMenuEntry("50x50", 25);
  glutAddMenuEntry("100x100", 50);
  glutAddMenuEntry("Save...", 0);
  
  // Create the map window.
  mapWin = glutCreateWindow(kmt_file.c_str());
  glutEntryFunc(enter_leave);
  init();
  glutDisplayFunc(mapWindowDisplay);
  glutReshapeFunc(mapWindowReshape);
  glutMotionFunc(mapWindowMotion);
  glutMouseFunc(mapWindowMouse);
  glutKeyboardFunc(keyboard);  

  glutSetMenu(mapMenuID);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  // Create submap window.
  zoomMapWin = glutCreateWindow("Zoom Window");
  glutEntryFunc(enter_leave);
  init();
  glutDisplayFunc(zoomWindowDisplay);
  glutReshapeFunc(zoomWindowReshape);
  glutMotionFunc(zoomWindowMotion);
  
  glutKeyboardFunc(keyboard);
  glutMouseFunc(zoomWindowMouse);

  glutSetMenu(mapMenuID);
  glutAttachMenu(GLUT_RIGHT_BUTTON);
  
  glutMainLoop();

  delete tData;
  
  return 0;
}
