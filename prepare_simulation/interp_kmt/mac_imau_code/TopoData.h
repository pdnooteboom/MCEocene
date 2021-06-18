
#ifndef TopoData_H_
#define TopoData_H_

#define MAPWINDOW_HEIGHT  480
#define MAPWINDOW_WIDTH   650

class TopoData
{
 public:
  TopoData(void);
  TopoData(unsigned int width, unsigned int height, const char *filename = 0);
  ~TopoData(void);

  inline bool isValid(void)
  { return valid; };

  int elevation(int x, int y);
  void setValue(int x, int y, int value);  
  unsigned char pixel(int x, int y);
  inline unsigned char *pixels(void)
  { return image; };
  inline unsigned int getDSWidth(void)
  { return ds_width; };
  inline unsigned int getDSHeight(void)
  { return ds_height; };  
  inline unsigned int getMapVPWidth(void)
  { return mapVPWidth; };
  inline unsigned int getMapVPHeight(void)
  { return mapVPHeight; };  
  inline void setMapVPWidth(unsigned int w)
  { mapVPWidth = w; };
  inline void setMapVPHeight(unsigned int h)
  { mapVPHeight = h; };  
  inline unsigned int getZoomVPWidth(void)
  { return zoomVPWidth; };
  inline unsigned int getZoomVPHeight(void)
  { return zoomVPHeight; };  
  inline void setZoomVPWidth(unsigned int w)
  { zoomVPWidth = w; };
  inline void setZoomVPHeight(unsigned int h)
  { zoomVPHeight = h; };  
  inline void setAspectRatio(float w, float h)
  { aspectRatio = w/h; };  
  inline float getAspectRatio(void)
  { return aspectRatio; };  
  
  int readKMT(const char *filename);     // Read a KMT file.
  int writeKMT(const char *filename);    // Write a new KMT file.
  int writeImage(const char *filename);  // Save the image data out.
  
 protected:
  void buildImage(void);
  
 private:
  bool                valid;             // Valid/allocated data set?
  unsigned int        ds_width, ds_height;  // Data Set Dimensions
  unsigned int        mapVPWidth, mapVPHeight;    // Map viewport dimensions.
  unsigned int        zoomVPWidth, zoomVPHeight;  // Zoom viewport dimensions.
  unsigned char      *image;             // Image version of data.
  int                *elev;              // Actual elevation data.
  int                 min, max;          // Min and max data points.
  float               aspectRatio;       // Aspect Ratio of the data set.
};


#endif

/* EOF */

  
