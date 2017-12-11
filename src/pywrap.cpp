#include "pywrap.h"


using namespace DBoW2;
using namespace DUtils;
using namespace std;


PyDBoW2::PyDBoW2()
{
	// branching factor and depth levels 
	const int k = 9;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType score = L1_NORM;

	voc = OrbVocabulary(k, L, weight, score);
	orb = cv::ORB::create();

}

// Borrowed from https://stackoverflow.com/questions/22667093/how-to-convert-the-numpy-ndarray-to-a-cvmat-using-python-c-api
cv::Mat PyDBoW2::pyObjToMat(PyObject* source)
{

	PyArrayObject* contig = (PyArrayObject*)PyArray_FromAny(source,
		PyArray_DescrFromType(NPY_UINT8),
		2, 2, NPY_ARRAY_CARRAY, NULL);
	if (!contig) 
		throw runtime_error("Only use 2d arrays");
	

	cv::Mat mat(PyArray_DIM(contig, 0), PyArray_DIM(contig, 1), CV_8UC1,
		    PyArray_DATA(contig));
	return mat;
}

void PyDBoW2::changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
	//out.resize(plain.rows);

  	for(int i = 0; i < plain.rows; ++i)
  		out[i] = plain.row(i);
}

vector<cv::Mat> PyDBoW2::getFeatures(PyObject* im)
{
	cv::Mat mask;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	orb->detectAndCompute(pyObjToMat(im), mask, keypoints, descriptors);
	
	vector<cv::Mat> features(descriptors.rows);
	changeStructure(descriptors, features);	
	return features;

}

void PyDBoW2::addVoc(PyObject* im)
{

	vector<cv::Mat> features = getFeatures(im);	
	voc_vec.push_back(features);
}

void PyDBoW2::createVocAndDB()
{
	// create vocab and database from the same images as vocab
	voc.create(voc_vec);
	db = OrbDatabase(voc, false, 0); // false = do not use direct index
	for (int i = 0; i < voc_vec.size(); i++)
		db.add(voc_vec[i]);
}

PyObject* PyDBoW2::queryResultsToPyTuple(const QueryResults &q) 
{
	PyObject* tuple = PyTuple_New( 2 );
	if (!tuple) throw logic_error("Unable to allocate memory for Python tuple");
	PyObject *id = PyInt_FromSize_t(static_cast<size_t>(q[0].Id));
	if (!id)
	{
		Py_DECREF(tuple);
		throw logic_error("Unable to allocate memory for Python tuple");
	}

	PyTuple_SET_ITEM(tuple, 0, id);

	PyObject *score = PyFloat_FromDouble(q[0].Score);
	if (!score)
	{
		Py_DECREF(tuple);
		throw logic_error("Unable to allocate memory for Python tuple");
	}
	PyTuple_SET_ITEM(tuple, 1, score);
	
	return tuple;
}


PyObject* PyDBoW2::getClosestMatch(PyObject* im)
{
	const vector<cv::Mat> features = getFeatures(im);	
	QueryResults q;
	db.query(features, q);
	return queryResultsToPyTuple(q);
}

boost::shared_ptr<PyDBoW2> initWrapper()
{
	boost::shared_ptr<PyDBoW2> ptr( new PyDBoW2 );
	return ptr;
}



#if PY_VERSION_HEX >= 0x03000000
void *
#else
void
#endif
initialize()
{
  import_array();
}
BOOST_PYTHON_MODULE(dbow2)
{
	initialize();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	boost::python::class_< PyDBoW2, boost::shared_ptr<PyDBoW2>, boost::noncopyable>("PyDBoW2", boost::python::no_init)
		.def("__init__", boost::python::make_constructor(&initWrapper))
		.def("addVoc", &PyDBoW2::addVoc)
		.def("getClosestMatch", &PyDBoW2::getClosestMatch)
		.def("createVocAndDB", &PyDBoW2::createVocAndDB)
	;
	
}




