//#include <Python.h>
//#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace DUtils;
using namespace std;

class PyDBoW2
{
private:

cv::Ptr<cv::ORB> orb;
OrbVocabulary voc;
OrbDatabase db; // false = do not use direct index

boost::python::tuple queryResultsToPyTuple(const QueryResults &q);
//cv::Mat pyObjToMat(PyObject* source);
cv::Mat pyObjToMat(const boost::python::numpy::ndarray &im);

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);

//vector<cv::Mat> getFeatures(PyObject* im);
vector<cv::Mat> getFeatures(const boost::python::numpy::ndarray &im);

public:

PyDBoW2(const std::string &vocab_file);

void addToDB(const boost::python::numpy::ndarray &im); // im will be converted to cv::Mat
boost::python::tuple getClosestMatch(const boost::python::numpy::ndarray &im); // Get the tuple of "index, score" for the matching of the new image 
};

boost::shared_ptr<PyDBoW2> initWrapper(const std::string &vocab_file);



