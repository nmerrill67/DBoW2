
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "boost/filesystem.hpp"
#include <ctime>
#include <stdlib.h>
#include <fstream>
#include <iterator>

/*************************************************************************
* This creates an executable to test query time for varying database size.
* Simply run:
* 	$ vary-db-size <mem dir> <live dir> 
* where <mem dir> is the directory containing the images to be saved into memory,
* and <live dir> is the directory containing the images of the same locations but with 
* altered viewpoints. You could easily just throw any two directories in here, since this does not 
* test accuracy.
**************************************************************************/ 
using namespace DBoW2;
using namespace DUtils;
using namespace std;

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
vector<cv::Mat> getFeatures(cv::Mat &im, cv::Ptr<cv::ORB> &orb);

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
	//out.resize(plain.rows);

  	for(int i = 0; i < plain.rows; ++i)
  		out[i] = plain.row(i);
}

vector<cv::Mat> getFeatures(cv::Mat &im, cv::Ptr<cv::ORB> &orb)
{
	cv::Mat mask;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	orb->detectAndCompute(im, mask, keypoints, descriptors);
	
	vector<cv::Mat> features(descriptors.rows);
	changeStructure(descriptors, features);	
	return features;
}


int main(int argc, char** argv)
{

	if (argc < 3)
	{
		std::cout << "Usage:\n\tspeed-test <mem dir> <live dir>\n";
		return -1;
	}
	bool rs = false; // Resize images or not
	if (argc == 4 && strcmp("-r", argv[3]) == 0)
	{
		std::cout << "Resizing images to 120x160\n";
		rs = true;
	}
	else if (argc == 4)
	{
		throw std::runtime_error("Unrecognized argument: " + std::string(argv[3]));
	}
	boost::filesystem::path mem(argv[1]);
	boost::filesystem::path live(argv[2]);
	boost::filesystem::directory_iterator end_itr; // NULL

	cv::Mat im;	
	cv::Size sz(160, 120);
	// branching factor and depth levels 
	const int k = 9;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType score = L1_NORM;

	std::cout << "Loading ORB Vocabulary ...\n";
	clock_t start = clock();
	OrbVocabulary voc(k, L, weight, score);
  	voc.loadFromTextFile("ORBvoc.txt");
	std::cout << "Time to load vocab tree = " << (double)(clock()-start)/CLOCKS_PER_SEC << "\n";
	cv::Ptr<cv::ORB> orb = cv::ORB::create();
	OrbDatabase db(voc, false, 0); // false = do not use direct index
	vector<cv::Mat> feats;
	list<vector<cv::Mat>> query_des;
	
	// Read live files and calculate descriptors only once
	for (boost::filesystem::directory_iterator itr2(live); itr2!=end_itr; ++itr2)
	{
		im = cv::imread(itr2->path().string(), cv::IMREAD_GRAYSCALE);
		if (rs)	
			cv::resize(im, im, sz);
		query_des.push_front(getFeatures(im, orb));
	}
	
	size_t m = 1, q = query_des.size(), n = 0;
	double query_t = 0.0;
	std::vector<double> times; // results 
	std::vector<size_t> db_sizes; // results 
	QueryResults qr;	
	for (boost::filesystem::directory_iterator itr1(mem); itr1!=end_itr; ++itr1)
	{
		n++;
		im = cv::imread(itr1->path().string(), cv::IMREAD_GRAYSCALE);
		if (rs)	
			cv::resize(im, im, sz);
		feats = getFeatures(im, orb);
		int id = db.add(feats);
		std::cout << "Added Image " << id << " to database\n\n";
	
		if (m%10==0)
		{
			std::cout << "Timing queries for database size=" << m << "\n";
			for (vector<cv::Mat> d : query_des)
			{	
				start = clock();
				db.query(d, qr); // query(im, false) means return 1 result in q, and don't add im's descriptor to database afterwards
				query_t += ((double)clock()-start)/CLOCKS_PER_SEC * 1000 ; // (ms)
			}
			times.push_back(query_t / q);
			db_sizes.push_back(m);
			query_t = 0.0;
		}	
		m++;
	}
	
	std::cout << "\n\n\tResults\n\n";
	std::cout << "bins = " << n / 10 << "\n";
	std::cout << "N = " << n << "\n";
	
	times.shrink_to_fit();
	db_sizes.shrink_to_fit();
	std::string f = "vary-db-size-dbow-results.txt";
	if (rs)
		f = "vary-db-size-dbow-120x160-results.txt";
	std::ofstream results_file(f);
	std::ostream_iterator<size_t> m_it(results_file, " ");
	std::ostream_iterator<double> t_it(results_file, " ");
	std::copy(db_sizes.begin(), db_sizes.end(), m_it);
	results_file << "\n";
	std::copy(times.begin(), times.end(), t_it);
	std::cout << "results written to " << f << "\n";
} 


