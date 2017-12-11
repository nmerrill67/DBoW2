
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
	db = OrbDatabase(voc, false, 0); // false = do not use direct index
	orb = cv::ORB::create();

}

// Borrowed from https://stackoverflow.com/questions/22667093/how-to-convert-the-numpy-ndarray-to-a-cvmat-using-python-c-api
cv::Mat PyDBoW2::pyObjToMat(PyObject* source)
{

	PyArrayObject* contig = (PyArrayObject*)PyArray_FromAny(source,
		PyArray_DescrFromType(NPY_UINT8),
		2, 2, NPY_ARRAY_CARRAY, NULL);
	if (contig == nullptr) {
	  // Throw an exception
	  return;
	}

	cv::Mat mat(PyArray_DIM(contig, 0), PyArray_DIM(contig, 1), CV_8UC1,
		    PyArray_DATA(contig));
	return mat;
}

void PyDBoW2::changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
	out.resize(plain.rows);

  	for(int i = 0; i < plain.rows; ++i)
  		out[i] = plain.row(i);
}

vector<cv::Mat> PyDBoW2::getFeatures(PyObject* im)
{
	cv::Mat mask;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	orb->detectAndCompute(pyObjToMat(im), mask, keypoints, descriptors);
	
	vector<cv::Mat> features();
	changeStructure(descriptors, features);	
	return features;

}

void PyDBoW2::addVoc(PyObject* im)
{

	vector<cv::Mat> features = getFeatures(im);	
	BowVector v;
	voc.transform(features, v);
	db.add(features);
}

PyObject* PyDBoW2::queryResultsToPyTuple(const QueryResults &q) 
{
	PyObject* tuple = PyTuple_New( 2 );
	if (!tuple) throw logic_error("Unable to allocate memory for Python tuple");

	PyObject *id = PyInt_FromLong((long)q.Id);
	if (!id)
	{
		Py_DECREF(tuple);
		throw logic_error("Unable to allocate memory for Python tuple");
	}

	PyTuple_SET_ITEM(tuple, 0, id);

	PyObject *score = PyFloat_FromDouble(q.Score);
	if (!score)
	{
		Py_DECREF(tuple);
		throw logic_error("Unable to allocate memory for Python tuple");
	}
	PyTuple_SET_ITEM(tuple, 1, score);
	
	return tuple;
}


void PyObject* PyDBoW2::getClosestMatch(PyObject* im)
{
	vector<cv::Mat> features = getFeatures(im);	
	QueryResults ret;
	db.query(features, ret);
	
	return queryResultsToPyTuple(ret);
}

boost::shared_ptr<PyDBoW2> initWrapper()
{
	boost::shared_ptr<PyDBoW2> ptr( new PyDBoW2 );
	return ptr;
}






