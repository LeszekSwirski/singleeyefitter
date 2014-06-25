#include <iostream>
#include <fstream>
#include <regex>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include <tbb/tbb.h>

#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <pupiltracker/pupiltracker.h>

#include <singleeyefitter/singleeyefitter.h>
#include <singleeyefitter/fun.h>
#include <singleeyefitter/projection.h>

#include "clipper.hpp"


using namespace singleeyefitter;

struct fitEyeModel_ret {
	Sphere<double> est_eye, est_eye_lm, est_eye_contrast;
	std::vector<Circle3D<double>> est_pupils, est_pupils2, est_pupils_lm, est_pupils_contrast;
};


namespace detail {
    using namespace boost::spirit;

    template<class T> struct as_qi;
    template<> struct as_qi<int> { static const int_type& get() {return int_;} };
    template<> struct as_qi<double> { static const double_type& get() {return double_;} };

    template<class To>
    struct parse_helper {
        template<class From>
        static To parse(const From& src) {
            using std::begin;
            using std::end;
            To ret;
            bool success = qi::parse(begin(src), end(src), as_qi<To>::get(), ret);
            if (!success) {
                throw std::runtime_error("Parse int failed");
            }
            return ret;
        }

        static To parse(char* src) {
            To ret;
            bool success = qi::parse(src, src + strlen(src), as_qi<To>::get(), ret);
            if (!success) {
                throw std::runtime_error("Parse int failed");
            }
            return ret;
        }

        static To parse(const char* src) {
            To ret;
            bool success = qi::parse(src, src + strlen(src), as_qi<To>::get(), ret);
            if (!success) {
                throw std::runtime_error("Parse int failed");
            }
            return ret;
        }
    };
}

template<typename T>
int parse_int(T&& src) {
    return ::detail::parse_helper<int>::parse(std::forward<T>(src));
}
template<typename T>
double parse_double(T&& src) {
    return ::detail::parse_helper<double>::parse(std::forward<T>(src));
}


cv::Point2f toImgCoord(const cv::Point2f& point, const cv::Mat& m, double scale = 1, int shift = 0) {
    return cv::Point2f(static_cast<float>((m.cols/2 + scale*point.x) * (1<<shift)),
        static_cast<float>((m.rows/2 + scale*point.y) * (1<<shift)));
}
cv::Point toImgCoord(const cv::Point& point, const cv::Mat& m, double scale = 1, int shift = 0) {
    return cv::Point(static_cast<int>((m.cols/2 + scale*point.x) * (1<<shift)),
        static_cast<int>((m.rows/2 + scale*point.y) * (1<<shift)));
}
cv::RotatedRect toImgCoord(const cv::RotatedRect& rect, const cv::Mat& m, float scale = 1) {
    return cv::RotatedRect(toImgCoord(rect.center,m,scale),
        cv::Size2f(scale*rect.size.width,
        scale*rect.size.height),
        rect.angle);
}

namespace boost {
	namespace serialization {

		template<class Archive, class T>
		inline void serialize(Archive & ar, Ellipse2D<T> & g, const unsigned int version)
		{
			ar & g.centre;
			ar & g.major_radius;
			ar & g.minor_radius;
			ar & g.angle;
		}
		template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
		inline void serialize(Archive & ar, 
			                  Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
			                  const unsigned int file_version) 
		{
			int rows = t.rows(), cols = t.cols();
			ar & rows;
			ar & cols;
			if( rows * cols != t.size() )
				t.resize( rows, cols );

			for(int i=0; i<t.size(); i++)
				ar & t.data()[i];
		}
		template<class Archive, class T>
		inline void serialize(Archive & ar, cv::Point_<T> & g, const unsigned int version)
		{
			ar & g.x;
			ar & g.y;
		}

	} // namespace serializationstd::vector<Eigen::Vector2d>
} // namespace boost

template <typename T>
std::vector<size_t> sort_indexes_impl(const T &v, std::random_access_iterator_tag)
{
	static_assert(std::is_same<typename std::iterator_traits<typename T::iterator>::iterator_category, std::random_access_iterator_tag>::value, "Type matches tag");
	
	// initialize original index locations
	auto idx = fun::range_<std::vector<size_t>>(v.size());

	// sort indexes based on comparing values in v
    sort(std::begin(idx), std::end(idx), [&](size_t i1, size_t i2){ return v[i1] < v[i2]; });

	return idx;
}

template <typename T>
std::vector<size_t> sort_indexes(const T &v)
{
	return sort_indexes_impl(v, typename std::iterator_traits<typename T::iterator>::iterator_category());
}

template< typename T >
void reorder(std::vector<T>& vals, const std::vector<size_t>& idxs)  {  
    vals = fun::map([&](size_t i){ return vals[i]; }, idxs);
}

struct PupilGroundTruth {
    Eigen::Vector3d gaze_vector;
    std::vector<Eigen::Vector2d> outline;
    PupilGroundTruth() {}
    PupilGroundTruth(
        Eigen::Vector3d gaze_vector,
        std::vector<Eigen::Vector2d> outline) :
            gaze_vector(std::move(gaze_vector)),
            outline(std::move(outline)) {}
};

double calcEllipseTruthOverlap(const Ellipse2D<double>& ellipse, const std::vector<Eigen::Vector2d>& truth_poly) {
    ClipperLib::Path truth_path;
    ClipperLib::Path ellipse_path;

    int N = truth_poly.size();
    for(int i = 0; i < N; ++i) {
        auto p = truth_poly[i];

        double theta = (double)i / N * 2 * PI;
        auto p2 = pointAlongEllipse(ellipse, -theta);

        truth_path.emplace_back(static_cast<int>(std::floor(p.x()*100+0.5)),
                                static_cast<int>(std::floor(p.y()*100+0.5)));
        ellipse_path.emplace_back(static_cast<int>(std::floor(p2.x()*100+0.5)),
                                  static_cast<int>(std::floor(p2.y()*100+0.5)));
    };

    ClipperLib::Clipper clpr;
    clpr.AddPolygon(truth_path, ClipperLib::ptSubject);
    clpr.AddPolygon(ellipse_path, ClipperLib::ptClip);

    ClipperLib::Paths intersection_paths, union_paths;
    clpr.Execute(ClipperLib::ctIntersection, intersection_paths);
    clpr.Execute(ClipperLib::ctUnion, union_paths);

    double intersection_area = fun::sum([](const ClipperLib::Path& path){ return ClipperLib::Area(path); }, intersection_paths);
    double union_area = fun::sum([](const ClipperLib::Path& path){ return ClipperLib::Area(path); }, union_paths);

    return intersection_area / union_area;
}

void writeResults(const std::string& filename,
                  const std::vector<int> ids,
                  const std::vector<EyeModelFitter::Pupil>& null_pupils,
                  const std::vector<EyeModelFitter::Pupil>& simple_pupils,
                  const std::vector<EyeModelFitter::Pupil>& contrast_pupils,
                  const std::vector<EyeModelFitter::Pupil>& edge_pupils,
                  const std::map<int, PupilGroundTruth>& true_pupils,
                  double focal_length) {
	std::ofstream of(filename);

	for (int i = 0; i < ids.size(); ++i) {
        const auto& pupil = null_pupils[i].circle;
        const auto& pupil2 = simple_pupils[i].circle;
        const auto& pupil_dlib = contrast_pupils[i].circle;
        const auto& pupil_lm = edge_pupils[i].circle;
        const auto& id = ids[i];
        auto true_pupil_it = true_pupils.find(id);
        if (true_pupil_it == true_pupils.end())
            continue;

        const auto& true_pupil = true_pupil_it->second;

        double anglediff = acos(pupil.normal.dot(true_pupil.gaze_vector))*180/PI;
        double anglediff2 = acos(pupil2.normal.dot(true_pupil.gaze_vector))*180/PI;
        double anglediff_dlib = acos(pupil_dlib.normal.dot(true_pupil.gaze_vector))*180/PI;
        double anglediff_lm = acos(pupil_lm.normal.dot(true_pupil.gaze_vector))*180/PI;

        auto true_pupil_outline = true_pupil.outline;

        double ellipdist = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil, focal_length)), true_pupil_outline);
        double ellipdist2 = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil2, focal_length)), true_pupil_outline);
        double ellipdist_dlib = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil_dlib, focal_length)), true_pupil_outline);
        double ellipdist_lm = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil_lm, focal_length)), true_pupil_outline);

		of << id;
		of << "\t" << anglediff << "\t" << anglediff2 << "\t" << anglediff_dlib << "\t" << anglediff_lm;
		of << "\t" << ellipdist << "\t" << ellipdist2 << "\t" << ellipdist_dlib << "\t" << ellipdist_lm;
		of << std::endl;
	}
}

std::string regex_escape(const std::string& pattern) {
    return regex_replace(pattern,
        std::regex("[\\^\\.\\$\\|\\(\\)\\[\\]\\*\\+\\?\\/\\\\]"),
        std::string("\\$&"));
}

struct Key {
    wchar_t code;
    bool shift;
    bool ctrl;
    bool meta;
};

#ifdef WIN32

Key cvxWaitKey(int delay) {
    int code = cv::waitKey(delay);
    Key key;
    key.code = code;
    key.shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
    key.ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
    key.meta = (GetKeyState(VK_MENU) & 0x8000) != 0;
    return key;
}

#else

Key cvxWaitKey(int delay) {
    int code = cv::waitKey(delay);
    Key key;
    if (code >= 'A' && code <= 'Z') {
        key.code = code + 'a' - 'A';
        key.shift = true;
    }
    else {
        key.code = code;
        key.shift = false;
    }
    key.ctrl = false;
    key.meta = false;
    return key;
}

#endif

int main(int argc, char* argv[])
{
    using boost::math::sign;

    if (argc < 4) {
        std::cerr << "USAGE: " << argv[0] << " <folder> <filepattern> <fov> [<truthfile> [<init_num_added>]]" << std::endl;
        return 1;
    }

	//try
	{
		namespace fs = boost::filesystem;

		std::vector<int> ids;
		std::vector<cv::Mat> obs_eye_images;
		std::vector<Ellipse2D<double>> obs_pupil_ellipses;
		std::vector<std::vector<cv::Point2f>> obs_pupil_inliers;

		fs::path imagedir(argv[1]);
        std::string filepattern(argv[2]);

        std::smatch filepatternisregex_match;
        if (regex_match(filepattern, filepatternisregex_match, std::regex("/(.*)/"))) {
            filepattern = filepatternisregex_match[1].str();
        } else {
            filepattern = regex_escape(filepattern);
            // Converts #### into (\d\d\d\d\d*)
            filepattern = regex_replace(filepattern, std::regex("#+"), std::string("($&\\d*)"));
            filepattern = regex_replace(filepattern, std::regex("#"), std::string("\\d"));
        }
		std::regex file_regex(filepattern);

		tbb::mutex push_lock;

		tbb::parallel_for_each(fs::directory_iterator(imagedir), fs::directory_iterator(), [&] (fs::path path) {
			std::smatch path_match;
			std::string path_str = path.filename().string();

			if (is_regular_file(path) && regex_match(path_str, path_match, file_regex)) {
				int id = parse_int(path_match[1].str());
				cv::Mat eye = cv::imread(path.string(), CV_LOAD_IMAGE_GRAYSCALE);
				Ellipse2D<double> el;
				std::vector<cv::Point2f> inlier_pts;

				bool cache_valid = false;
				const int CACHE_VERSION = 8;

				fs::path cache_path = path.parent_path() / fs::path(path.stem().string() + ".cache");

				if (exists(cache_path)) {
					std::ifstream ifs(cache_path.string());
					boost::archive::text_iarchive ia(ifs);

					int ia_cache_version;
					ia >> ia_cache_version;
					if (CACHE_VERSION == ia_cache_version) {
						std::cout << "Loading " << cache_path.string() << std::endl;

						ia >> el;
						ia >> inlier_pts;

						cache_valid = true;
					}
				}

				if (!cache_valid)
				{
					std::cout << "Pupil tracking " << path.string() << std::endl;

					pupiltracker::tracker_log log;

					pupiltracker::TrackerParams pupil_tracker_params;
					pupil_tracker_params.Radius_Min = 20;
					pupil_tracker_params.Radius_Max = 70;
					pupil_tracker_params.CannyThreshold1 = 20;
					pupil_tracker_params.CannyThreshold2 = 40;
					pupil_tracker_params.CannyBlur = 1.6;
					pupil_tracker_params.EarlyRejection = true;
					pupil_tracker_params.EarlyTerminationPercentage = 95;
					pupil_tracker_params.PercentageInliers = 20;
					pupil_tracker_params.InlierIterations = 2;
					pupil_tracker_params.ImageAwareSupport = true;
					pupil_tracker_params.StarburstPoints = 0;
					//pupil_tracker_params.Seed = 0;

					pupiltracker::findPupilEllipse_out pupil_tracker_out;
					bool found = pupiltracker::findPupilEllipse(pupil_tracker_params, eye, pupil_tracker_out, log);

					if (found) {
						el = toEllipse<double>(pupil_tracker_out.elPupil);
						el.centre -= Eigen::Vector2d(eye.cols, eye.rows)/2;

						for (auto&& inlier : pupil_tracker_out.inliers) {
							inlier_pts.push_back(cv::Point2f(pupil_tracker_out.roiPupil.x + inlier.x - eye.cols/2, pupil_tracker_out.roiPupil.y + inlier.y - eye.rows/2));
						}
					} else {
						el = Ellipse2D<double>::Null;
					}

					std::ofstream ofs(cache_path.string());
					boost::archive::text_oarchive oa(ofs);

					oa << CACHE_VERSION;
					oa << el;
					oa << inlier_pts;
				}

				{
					tbb::mutex::scoped_lock push_guard(push_lock);

					if (el) {
						ids.push_back(id);
						obs_eye_images.push_back(eye);
						obs_pupil_ellipses.push_back(el);
						obs_pupil_inliers.push_back(std::move(inlier_pts));
					}
				}
			}
		});

		auto&& idx_sort = sort_indexes(ids);
		reorder(ids, idx_sort);
		reorder(obs_eye_images, idx_sort);
		reorder(obs_pupil_ellipses, idx_sort);
		reorder(obs_pupil_inliers, idx_sort);

        double fov = parse_double(argv[3]);
		fov *= PI / 180;
		double focal_length = (obs_eye_images[0].cols / 2) / std::tan(fov/2);

		bool animate = false;


		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// This is where the stuff actually happens
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        EyeModelFitter null_fitter(focal_length, 5, 0.5);
        EyeModelFitter simple_fitter(focal_length, 5, 0.5);
        EyeModelFitter contrast_fitter(focal_length, 5, 0.5);
        EyeModelFitter edge_fitter(focal_length, 5, 0.5);

        auto N = obs_eye_images.size();
        if (argc > 5) {
            int n = parse_int(argv[5]);
            if (n >= 0 && n < N) {
                N = n;
            }
        }

        for (int i = 0; i < N; ++i) {
            null_fitter.add_observation(obs_eye_images[i], obs_pupil_ellipses[i], obs_pupil_inliers[i]);
            simple_fitter.add_observation(obs_eye_images[i], obs_pupil_ellipses[i], obs_pupil_inliers[i]);
            contrast_fitter.add_observation(obs_eye_images[i], obs_pupil_ellipses[i], obs_pupil_inliers[i]);
            edge_fitter.add_observation(obs_eye_images[i], obs_pupil_ellipses[i], obs_pupil_inliers[i]);
        }

        null_fitter.unproject_observations();

        simple_fitter.unproject_observations();
        simple_fitter.initialise_model();

        contrast_fitter.unproject_observations();
        contrast_fitter.initialise_model();

        edge_fitter.unproject_observations();
        edge_fitter.initialise_model();

        Sphere<double> true_eye;
        std::map<int, PupilGroundTruth> true_pupils;

        if (argc > 4) {
            using namespace boost::spirit;
            using boost::phoenix::at_c;

            std::ifstream ground_truth_file((imagedir / argv[4]).string());
            if (ground_truth_file.is_open()) {
                std::string str;
                std::getline(ground_truth_file, str);
		        while(ground_truth_file.good()) {
			        int id;
			        PupilGroundTruth true_pupil;

                    /* One day I will understand boost qi enough to do something like this:

                    qi::rule<decltype(begin(str)), Eigen::Vector2d(), qi::ascii::space_type> vec2d =
                        qi::eps[_val = Eigen::Vector2d()] >>
                        qi::double_[at_c<0>(_val) = _1] >> ',' >>
                        qi::double_[at_c<1>(_val) = _1];
                    qi::rule<decltype(begin(str)), Eigen::Vector3d(), qi::ascii::space_type> vec3d =
                        qi::eps[qi::_val = Eigen::Vector3d()] >>
                        qi::double_[at_c<0>(_val) = _1] >> ',' >>
                        qi::double_[at_c<1>(_val) = _1] >> ',' >>
                        qi::double_[at_c<2>(_val) = _1];

                    But it is not that day */

                    std::vector<std::pair<double,double>> outlinepoints;

                    int parse_success = qi::phrase_parse(begin(str), end(str),
                        int_ >> '|' >>
                        double_ >> ',' >> double_ >> ',' >> double_ >> '|' >>
                        double_ >> '|' >>
                        double_ >> ',' >> double_ >> ',' >> double_ >> '|' >>
                        (double_ >> ',' >> double_) % ',',
                        ascii::space,
                        id,
                        true_eye.centre[0],true_eye.centre[1],true_eye.centre[2],
                        true_eye.radius,
                        true_pupil.gaze_vector[0],true_pupil.gaze_vector[1],true_pupil.gaze_vector[2],
                        outlinepoints
                    );
                    if (parse_success) {
                        true_eye.centre[1] = -true_eye.centre[1];
                        true_eye.centre[2] = -true_eye.centre[2];
                        true_pupil.gaze_vector[1] = -true_pupil.gaze_vector[1];
                        true_pupil.gaze_vector[2] = -true_pupil.gaze_vector[2];
                        if (std::find(begin(ids), end(ids), id) != ids.end()) {
                            for (const auto& p : outlinepoints) {
                                true_pupil.outline.emplace_back(p.first, -p.second);
                            }
                            true_pupils[id] = true_pupil;
                        }

                    } else {
                        std::cout << "Failed to parse ground truth: " << str << std::endl;
                    }

			        std::getline(ground_truth_file, str);
		        }
            }
        }


		double displayscale = 1;
		bool do_display = true;
		int curr_i = 0;

		namespace acc=boost::accumulators;
		acc::accumulator_set<double, acc::stats<acc::tag::mean, acc::tag::variance>> acc_ellipdist, acc_ellipdist2, acc_ellipdist_dlib, acc_ellipdist_lm;
		bool recalc_ellipdist = true;

		bool display_obs_pupil_ellipses = true;
		bool display_est_pupil_ellipses = true;
		bool display_est_pupil_ellipses_contrast = true;
		bool display_est_pupil_ellipses_lm = true;
		bool display_true_pupil_ellipses = true;
		bool display_intersection_ellip = false;
        bool display_intersection_lines = false;

        bool ransac = true;
        bool use_smoothness = true;

        auto animate_func = 
            [&] (const Sphere<double>& eye, const std::vector<Circle3D<double>>& pupils) {
                cv::Mat curr = obs_eye_images[curr_i];
                cv::Mat curr_disp = cvx::resize(cvx::cvtColor(cvx::convert(curr, CV_8U, 255),
                                                              cv::COLOR_GRAY2BGR),
                                                displayscale, 0, cv::INTER_CUBIC);

                cv::ellipse(curr_disp, toImgCoord(toRotatedRect(project(eye, focal_length)), curr_disp, displayscale), cvx::rgb(60,0,0), 1, CV_AA);
                for (const auto& pupil : pupils) {
                    if (pupil)
                        cv::ellipse(curr_disp, toImgCoord(toRotatedRect(Ellipse2D<double>(project(pupil, focal_length))), curr_disp, displayscale), cvx::rgb(60,60,0), 1, CV_AA);
                }
                if (curr_i < pupils.size() && pupils[curr_i]) {
                    cv::ellipse(curr_disp, toImgCoord(toRotatedRect(Ellipse2D<double>(project(pupils[curr_i], focal_length))), curr_disp, displayscale), cvx::rgb(60,0,0), 1, CV_AA);
                }
                cv::imshow("Current Eye", curr_disp);
                cv::waitKey(10);
            };

        //cv::namedWindow("Eye");
        //cv::createButton("Display observed ellipses", [](int state, void* userdata){*(bool*)userdata = state;}, &display_obs_pupil_ellipses, cv::QT_CHECKBOX, display_obs_pupil_ellipses);
        //cv::createButton("Display initial pupil model", [](int state, void* userdata){*(bool*)userdata = state;}, &display_est_pupil_ellipses, cv::QT_CHECKBOX, display_est_pupil_ellipses);
        //cv::createButton("Display contrast-optimized pupils", [](int state, void* userdata){*(bool*)userdata = state;}, &display_est_pupil_ellipses_contrast, cv::QT_CHECKBOX, display_est_pupil_ellipses_contrast);

		cv::VideoWriter recording_writer;
		while(do_display) {
            cv::Mat disp = cv::Mat::zeros(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_8UC3);

			
            cv::Mat curr = obs_eye_images[curr_i];
            cv::Mat curr_disp = cvx::resize(cvx::cvtColor(curr,
                                                          cv::COLOR_GRAY2BGR),
                                            displayscale, 0, cv::INTER_CUBIC);

			cv::Mat curr_edge_disp = cv::Mat::zeros(curr_disp.rows, curr_disp.cols, CV_8UC3);

			for (const auto& inlier : obs_pupil_inliers[curr_i]) {
				int x = std::floor(curr_edge_disp.cols/2 + inlier.x * displayscale - 0.5);
				int y = std::floor(curr_edge_disp.rows/2 + inlier.y * displayscale - 0.5);
				if (x < 0 || x >= curr_edge_disp.cols || y < 0 || y >= curr_edge_disp.rows)
					continue;

				curr_edge_disp.at<cv::Vec3b>(y,x) = cv::Vec3b::all(255);
			}

			if (display_true_pupil_ellipses) {
                if (true_eye) {
				    cv::ellipse(disp, toImgCoord(toRotatedRect(project(true_eye, focal_length)), disp, displayscale), cvx::rgb(60,60,60), -1, CV_AA);
                }
			}

			if (display_est_pupil_ellipses) {
                if (simple_fitter.eye) {
				    cv::ellipse(disp, toImgCoord(toRotatedRect(project(simple_fitter.eye, focal_length)), disp, displayscale), cvx::rgb(60,0,60), 1, CV_AA);
				    cv::ellipse(curr_disp, toImgCoord(toRotatedRect(project(simple_fitter.eye, focal_length)), curr_disp, displayscale), cvx::rgb(60,0,60), 1, CV_AA);
                }
			}
			if (display_est_pupil_ellipses_contrast) {
                if (contrast_fitter.eye) {
				    cv::ellipse(disp, toImgCoord(toRotatedRect(project(contrast_fitter.eye, focal_length)), disp, displayscale), cvx::rgb(0,60,0), 1, CV_AA);
				    cv::ellipse(curr_disp, toImgCoord(toRotatedRect(project(contrast_fitter.eye, focal_length)), curr_disp, displayscale), cvx::rgb(0,60,0), 1, CV_AA);
                }
			}
			if (display_est_pupil_ellipses_lm) {
                if (edge_fitter.eye) {
				    cv::ellipse(disp, toImgCoord(toRotatedRect(project(edge_fitter.eye, focal_length)), disp, displayscale), cvx::rgb(0,60,60), 1, CV_AA);
				    cv::ellipse(curr_disp, toImgCoord(toRotatedRect(project(edge_fitter.eye, focal_length)), curr_disp, displayscale), cvx::rgb(0,60,60), 1, CV_AA);
                }
			}
            if (display_true_pupil_ellipses) {
                if (true_eye) {
				    cv::ellipse(curr_disp, toImgCoord(toRotatedRect(project(true_eye, focal_length)), curr_disp, displayscale), cvx::rgb(60,0,0), 1, CV_AA);
                }
			}

			auto display_from = [&] (const std::vector<EyeModelFitter::Pupil>& pupils, double r, double g, double b) {
				for (const auto& pupil : pupils) {
					if (pupil.circle)
						cv::ellipse(disp, toImgCoord(toRotatedRect(Ellipse2D<double>(project(pupil.circle, focal_length))), disp, displayscale), cvx::rgb(r,g,b), 1, CV_AA);
				}
				if (curr_i < pupils.size() && pupils[curr_i].circle) {
					cv::ellipse(curr_disp, toImgCoord(toRotatedRect(Ellipse2D<double>(project(pupils[curr_i].circle, focal_length))), curr_disp, displayscale), cvx::rgb(r,g,b), 1, CV_AA);
					cv::ellipse(curr_edge_disp, toImgCoord(toRotatedRect(Ellipse2D<double>(project(pupils[curr_i].circle, focal_length))), curr_edge_disp, displayscale), cvx::rgb(r,g,b), 1, CV_AA);
				}
			};

			if (display_obs_pupil_ellipses) {
				display_from(null_fitter.pupils, 255,0,255);
			}
			if (display_est_pupil_ellipses) {
				display_from(simple_fitter.pupils, 255,255,0);
			}
			if (display_est_pupil_ellipses_contrast) {
				display_from(contrast_fitter.pupils, 0,255,0);
			}
			if (display_est_pupil_ellipses_lm) {
				display_from(edge_fitter.pupils, 0,255,255);
			}
			if (display_true_pupil_ellipses) {
                std::vector<std::vector<cv::Point>> pts;
                for (auto&& id : ids) {
                    const auto& true_pupils_id_it = true_pupils.find(id);
					if (true_pupils_id_it != true_pupils.end()) {
                        const auto& true_pupil = true_pupils_id_it->second;

                        auto this_pts = fun::map([&](const Eigen::Vector2d& pt){
                            auto imgcoord = toImgCoord(cv::Point2f(pt.x(), pt.y()), disp, displayscale, 5);
                            return cv::Point(imgcoord.x, imgcoord.y);
                        }, true_pupil.outline);
                        pts.emplace_back(std::move(this_pts));
                    }
				}
                cv::polylines(disp, pts, true, cvx::rgb(255,0,0), 1, CV_AA, 5);

                const auto& true_pupils_curri_it = true_pupils.find(ids[curr_i]);
                if (true_pupils_curri_it != true_pupils.end()) {
                    const auto& true_pupil = true_pupils_curri_it->second;

                    auto this_pts = fun::map([&](const Eigen::Vector2d& pt){
                        auto imgcoord = toImgCoord(cv::Point2f(pt.x(), pt.y()), curr_disp, displayscale, 5);
                        return cv::Point(imgcoord.x, imgcoord.y);
                    }, true_pupil.outline);
                    cv::polylines(curr_disp, this_pts, true, cvx::rgb(255,0,0), 1, CV_AA, 5);
                    cv::polylines(curr_edge_disp, this_pts, true, cvx::rgb(255,0,0), 1, CV_AA, 5);
				}
			}

			double anglediff_i, anglediff2_i, anglediff_dlib_i, anglediff_lm_i;
			double ellipdist_i, ellipdist2_i, ellipdist_dlib_i, ellipdist_lm_i;

			acc::accumulator_set<double, acc::stats<acc::tag::mean, acc::tag::variance>> acc_anglediff, acc_anglediff2, acc_anglediff_dlib, acc_anglediff_lm;
			if (recalc_ellipdist) {
				acc_ellipdist = acc::accumulator_set<double, acc::stats<acc::tag::mean, acc::tag::variance>>();
				acc_ellipdist2 = acc::accumulator_set<double, acc::stats<acc::tag::mean, acc::tag::variance>>();
				acc_ellipdist_dlib = acc::accumulator_set<double, acc::stats<acc::tag::mean, acc::tag::variance>>();
				acc_ellipdist_lm = acc::accumulator_set<double, acc::stats<acc::tag::mean, acc::tag::variance>>();
			}
			for (int i = 0; i < N; ++i) {
				const EyeModelFitter::Circle& pupil = null_fitter.pupils[i].circle;
				const EyeModelFitter::Circle& pupil2 = simple_fitter.pupils[i].circle;
				const EyeModelFitter::Circle& pupil_dlib = contrast_fitter.pupils[i].circle;
				const EyeModelFitter::Circle& pupil_lm = edge_fitter.pupils[i].circle;
				const auto& id = ids[i];
                auto true_pupil_it = true_pupils.find(id);
				if (true_pupil_it == true_pupils.end())
					continue;

				const auto& true_pupil = true_pupil_it->second;

				double anglediff = acos(pupil.normal.dot(true_pupil.gaze_vector))*180/PI;
				double anglediff2 = acos(pupil2.normal.dot(true_pupil.gaze_vector))*180/PI;
				double anglediff_dlib = acos(pupil_dlib.normal.dot(true_pupil.gaze_vector))*180/PI;
				double anglediff_lm = acos(pupil_lm.normal.dot(true_pupil.gaze_vector))*180/PI;

				if (i == curr_i) {
					anglediff_i = anglediff;
					anglediff2_i = anglediff2;
					anglediff_dlib_i = anglediff_dlib;
					anglediff_lm_i = anglediff_lm;
				}

				acc_anglediff(anglediff);
				acc_anglediff2(anglediff2);
				acc_anglediff_dlib(anglediff_dlib);
				acc_anglediff_lm(anglediff_lm);

				auto true_pupil_outline = true_pupil.outline;

				if (recalc_ellipdist || i == curr_i) {
					double ellipdist = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil, focal_length)), true_pupil_outline);
					double ellipdist2 = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil2, focal_length)), true_pupil_outline);
					double ellipdist_dlib = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil_dlib, focal_length)), true_pupil_outline);
					double ellipdist_lm = calcEllipseTruthOverlap(Ellipse2D<double>(project(pupil_lm, focal_length)), true_pupil_outline);

					if (i == curr_i) {
						ellipdist_i = ellipdist;
						ellipdist2_i = ellipdist2;
						ellipdist_dlib_i = ellipdist_dlib;
						ellipdist_lm_i = ellipdist_lm;
					}
					if (recalc_ellipdist) {
						acc_ellipdist(ellipdist);
						acc_ellipdist2(ellipdist2);
						acc_ellipdist_dlib(ellipdist_dlib);
						acc_ellipdist_lm(ellipdist_lm);
					}
				}
			}
			recalc_ellipdist = false;

			std::stringstream curr_i_ss;
			curr_i_ss << std::fixed << std::setprecision(4);
			curr_i_ss << "obs " << anglediff_i << ", ";
			curr_i_ss << "est " << anglediff2_i << ", ";
			curr_i_ss << "grd " << anglediff_dlib_i << ", ";
			curr_i_ss << "lm " << anglediff_lm_i << " | ";
			curr_i_ss << "obs " << ellipdist_i << ", ";
			curr_i_ss << "est " << ellipdist2_i << ", ";
			curr_i_ss << "grd " << ellipdist_dlib_i << ", ";
			curr_i_ss << "lm " << ellipdist_lm_i << " | ";
			curr_i_ss << "Frame " << curr_i << " (total " << obs_eye_images.size() << ")";

			std::stringstream ss;
			ss << std::fixed << std::setprecision(4);
			ss << "obs " << acc::mean(acc_anglediff) << " (" << acc::variance(acc_anglediff) << "), ";
			ss << "est " << acc::mean(acc_anglediff2) << " (" << acc::variance(acc_anglediff2) << "), ";
			ss << "grd " << acc::mean(acc_anglediff_dlib) << " (" << acc::variance(acc_anglediff_dlib) << "), ";
			ss << "lm " << acc::mean(acc_anglediff_lm) << " (" << acc::variance(acc_anglediff_lm) << ") | ";
			ss << "obs " << acc::mean(acc_ellipdist) << " (" << acc::variance(acc_ellipdist) << "), ";
			ss << "est " << acc::mean(acc_ellipdist2) << " (" << acc::variance(acc_ellipdist2) << "), ";
			ss << "grd " << acc::mean(acc_ellipdist_dlib) << " (" << acc::variance(acc_ellipdist_dlib) << "), ";
			ss << "lm " << acc::mean(acc_ellipdist_lm) << " (" << acc::variance(acc_ellipdist_lm) << ")";


			cv::imshow("Eye", disp);
			cv::imshow("Current Eye", curr_disp);
			cv::imshow("Current Eye Edges", curr_edge_disp);

			cv::displayStatusBar("Eye", ss.str());
			cv::displayStatusBar("Current Eye", curr_i_ss.str());

			if (recording_writer.isOpened()) {
				recording_writer << curr_disp;
			}

            if (display_intersection_ellip || display_intersection_lines) {
                cv::Mat disp_invalid_lines = cv::Mat::zeros(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_32FC1);
				cv::Mat disp_valid_lines = cv::Mat::zeros(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_32FC1);
				cv::Mat disp_ellipses = cv::Mat::zeros(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_32FC1);
				for (const auto& pupil : null_fitter.pupils) {
					Eigen::Matrix<double,2,1> c_proj = project(pupil.circle.centre, focal_length);
					Eigen::Matrix<double,2,1> v_proj = project(pupil.circle.centre+pupil.circle.normal, focal_length) - c_proj;
					Eigen::Matrix<double,2,1> start = c_proj - 500*v_proj;
					Eigen::Matrix<double,2,1> end = c_proj + 500*v_proj;

					if (display_intersection_lines) {

						cv::Mat disp_line = cv::Mat::zeros(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_8UC1);
						cvx::line(disp_line,
                            cv::Point2f(disp_line.cols/2 + displayscale*start[0], disp_line.rows/2 + displayscale*start[1]),
							cv::Point2f(disp_line.cols/2 + displayscale*end[0], disp_line.rows/2 + displayscale*end[1]),
							cv::Scalar(25));

						cv::Mat disp_line_f = cvx::convert(disp_line, CV_32FC1, 1./255, 0);

                        if (pupil.init_valid)
						    cv::accumulate((disp_line_f).mul(1 - disp_valid_lines), disp_valid_lines);
                        else
                            cv::accumulate((disp_line_f).mul(1 - disp_invalid_lines), disp_invalid_lines);
					}

					if (display_intersection_ellip) {
						cv::Mat disp_ellipse = cv::Mat::zeros(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_8UC1);
						cv::ellipse(disp_ellipse, toImgCoord(toRotatedRect(Ellipse2D<double>(project(pupil.circle, focal_length))), disp, displayscale), cv::Scalar(50), 1, CV_AA);

						cv::Mat disp_ellipse_f = cvx::convert(disp_ellipse, CV_32FC1, 1./255, 0);
						cv::accumulate((disp_ellipse_f).mul(1 - disp_ellipses), disp_ellipses);
					}
                }

                cv::Mat disp_intersection = cv::Mat(displayscale*obs_eye_images[0].rows, displayscale*obs_eye_images[0].cols, CV_32FC3, cvx::rgb(1,1,1));
                cv::accumulate(cvx::cvtColor(disp_invalid_lines, cv::COLOR_GRAY2BGR).mul(cvx::rgb(1,0,0) - disp_intersection), disp_intersection);
				cv::accumulate(cvx::cvtColor(disp_valid_lines, cv::COLOR_GRAY2BGR).mul(cvx::rgb(0,0,1) - disp_intersection), disp_intersection);
				cv::accumulate(cvx::cvtColor(disp_ellipses, cv::COLOR_GRAY2BGR).mul(cvx::rgb(1,0,1) - disp_intersection), disp_intersection);
				
                if (display_intersection_lines) {
                    auto crosspt = toImgCoord(toPoint2f(project(simple_fitter.eye.centre, focal_length)), disp, displayscale);
                    for (int i = -1; i <= 1; ++i)
                        for (int j = -1; j <= 1; ++j)
                            cvx::cross(disp_intersection, crosspt + cv::Point2f(i,j), 10, cvx::rgb(0,0,0), 3);
					cvx::cross(disp_intersection, crosspt, 10, cvx::rgb(0,200,0), 3);
				}

				cv::imshow("Intersection", disp_intersection);
			}

			Key key = cvxWaitKey(-1);

            switch(key.code) {
            case 16: // Shift
                break;

            case '1':
                display_obs_pupil_ellipses = !display_obs_pupil_ellipses;
                break;
            case '2':
                display_est_pupil_ellipses = !display_est_pupil_ellipses;
                break;
            case '3':
                display_est_pupil_ellipses_contrast = !display_est_pupil_ellipses_contrast;
                break;
            case '4':
                display_est_pupil_ellipses_lm = !display_est_pupil_ellipses_lm;
                break;
            case '5':
                display_true_pupil_ellipses = !display_true_pupil_ellipses;
                break;

			case 27: // ESC
			case'q':
				do_display = false;
                break;
            case 'w':
                writeResults((imagedir / "result_analysis.txt").string(),
                    ids,
                    null_fitter.pupils,
                    simple_fitter.pupils,
                    contrast_fitter.pupils,
                    edge_fitter.pupils,
                    true_pupils,
                    focal_length);
                break;

			case '+':
			case '=':
				displayscale *= 1.1;
				break;
			case '-':
			case '_':
				displayscale /= 1.1;
				break;
			case 'i':
				display_intersection_ellip = !display_intersection_ellip;
				break;
			case 'o':
				display_intersection_lines = !display_intersection_lines;
				break;

            case 'a': {
                if (key.shift) {

                    animate = !animate;
                    if (animate) {
                        std::cout << "Animation enabled" << std::endl;
                    } else {
                        std::cout << "Animation disabled" << std::endl;
                    }
                    
                } else {
                    int id;
                    if (N < obs_eye_images.size() - 1) {
                        id = null_fitter.add_observation(obs_eye_images[N], obs_pupil_ellipses[N], obs_pupil_inliers[N]);
                        null_fitter.unproject_single_observation(id);

                        id = simple_fitter.add_observation(obs_eye_images[N], obs_pupil_ellipses[N], obs_pupil_inliers[N]);
                        simple_fitter.unproject_single_observation(id);
                        simple_fitter.initialise_single_observation(id);

                        id = contrast_fitter.add_observation(obs_eye_images[N], obs_pupil_ellipses[N], obs_pupil_inliers[N]);
                        contrast_fitter.unproject_single_observation(id);
                        contrast_fitter.initialise_single_observation(id);
                        contrast_fitter.refine_single_with_contrast(id);

                        id = edge_fitter.add_observation(obs_eye_images[N], obs_pupil_ellipses[N], obs_pupil_inliers[N]);
                        edge_fitter.unproject_single_observation(id);
                        edge_fitter.initialise_single_observation(id);
                        //edge_fitter.refine_single_with_inliers(id);

                        curr_i = N;
                        N++;
                    }
                }
				break;
            }

            case 's':
                if (key.shift) {
                    for (int i = 0; i < N; ++i) {
                        std::cout << i << std::endl;
                        contrast_fitter.refine_single_with_contrast(i);
                    }
                    std::cout << "Done" << std::endl;
                }
                else {
                    use_smoothness = !use_smoothness;
                    if (use_smoothness) {
                        std::cout << "Smoothness enabled" << std::endl;
                    } else {
                        std::cout << "Smoothness disabled" << std::endl;
                    }
                }

                break;

            case 'd':
                if (key.shift) {
                    contrast_fitter.refine_single_with_contrast(curr_i);
                    contrast_fitter.print_single_contrast_metric(curr_i);
                }
                else {
                    if (animate)
                        contrast_fitter.refine_with_region_contrast(animate_func);
                    else
                        contrast_fitter.refine_with_region_contrast();
                    std::cout << "Done" << std::endl;
                }
                recalc_ellipdist = true;
                display_est_pupil_ellipses_contrast = true;

                break;

            case 'g':
                if (key.shift)
                {
                    std::string out_file = (imagedir / "contrast_sweep.txt").string();
                    std::ofstream out_stream(out_file);

                    auto start_params = contrast_fitter.pupils[curr_i].params;

                    for (auto&& theta : fun::linspace(0.0, PI, 100)) {
                        for (auto&& psi : fun::linspace(0.0, 2*PI, 100)) {
                            for (auto&& radius : fun::range_<std::vector<double>>(1, 10, 0.1)) {
                                EyeModelFitter::PupilParams params(theta, psi, radius);
                                contrast_fitter.pupils[curr_i].params = params;
                                std::cout << theta << " " << psi << " " << radius;
                                auto val = contrast_fitter.single_contrast_metric(curr_i);
                                std::cout << " -> " << val;

                                out_stream << theta << " " << psi << " " << radius << " " << val;
                            }
                        }
                    }

                    contrast_fitter.pupils[curr_i].params = start_params;
                    
                }
                else
                {
                    auto start_params = contrast_fitter.pupils[curr_i].params;
                
                    auto best_params = start_params;
                    double best_contrast = contrast_fitter.single_contrast_metric(curr_i);

                    for (auto&& theta : fun::linspace(start_params.theta - 0.1, start_params.theta + 0.1, 6)) {
                        for (auto&& psi : fun::linspace(start_params.psi - 0.1, start_params.psi + 0.1, 6)) {
                            for (auto&& radius : fun::linspace(start_params.radius / 1.5, start_params.radius * 1.5, 6)) {
                                EyeModelFitter::PupilParams params(theta, psi, radius);
                                contrast_fitter.pupils[curr_i].params = params;
                                std::cout << theta-start_params.theta << " " << psi-start_params.psi << " " << radius-start_params.radius;
                                contrast_fitter.refine_single_with_contrast(curr_i);
                                auto val = contrast_fitter.single_contrast_metric(curr_i);
                                std::cout << " -> " << val;
                                if (best_contrast >= val)
                                {
                                    best_params = contrast_fitter.pupils[curr_i].params;
                                    best_contrast = val;
                                    std::cout << " *";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }

                    contrast_fitter.pupils[curr_i].params = best_params;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(best_params);

                    recalc_ellipdist = true;
                    display_est_pupil_ellipses_contrast = true;
                }

                break;

            case '[':
                contrast_fitter.region_band_width--;
                std::cout << "Band width = " << contrast_fitter.region_band_width << std::endl;
                break;

            case ']':
                contrast_fitter.region_band_width++;
                std::cout << "Band width = " << contrast_fitter.region_band_width << std::endl;
                break;

            case ',':
                contrast_fitter.region_scale/=1.125;
                std::cout << "Region scale = " << contrast_fitter.region_scale << std::endl;
                break;

            case '.':
                contrast_fitter.region_scale*=1.125;
                std::cout << "Region scale = " << contrast_fitter.region_scale << std::endl;
                break;

            case 'n':
                null_fitter.unproject_observations(1, 20, ransac);

                simple_fitter.unproject_observations(1, 20, ransac);
                simple_fitter.initialise_model();

                contrast_fitter.unproject_observations(1, 20, ransac);
                contrast_fitter.initialise_model();

                edge_fitter.unproject_observations(1, 20, ransac);
                edge_fitter.initialise_model();

                std::cout << "Model reset" << std::endl;

                break;


            case 'h':
                if (key.shift) {
                    contrast_fitter.pupils[curr_i].params.psi -= 0.01;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(contrast_fitter.pupils[curr_i].params);
                    std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                    contrast_fitter.print_single_contrast_metric(curr_i);
                } else {
                    curr_i = curr_i == 0 ? obs_eye_images.size() - 1 : curr_i - 1;
                }
                break;
            case 'j':
                if (key.shift) {
                    contrast_fitter.pupils[curr_i].params.theta -= 0.01;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(contrast_fitter.pupils[curr_i].params);
                    std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                    contrast_fitter.print_single_contrast_metric(curr_i);
                }
                break;
            case 'k':
                if (key.shift) {
                    contrast_fitter.pupils[curr_i].params.theta += 0.01;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(contrast_fitter.pupils[curr_i].params);
                    std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                    contrast_fitter.print_single_contrast_metric(curr_i);
                }
                break;
            case 'l':
                if (key.shift) {
                    contrast_fitter.pupils[curr_i].params.psi += 0.01;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(contrast_fitter.pupils[curr_i].params);
                    std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                    contrast_fitter.print_single_contrast_metric(curr_i);
                } else {
				    curr_i = (curr_i + 1) % obs_eye_images.size();
                }
                break;
            case ';':
                if (key.shift) {
                    contrast_fitter.pupils[curr_i].params.radius /= 1.1;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(contrast_fitter.pupils[curr_i].params);
                    std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                    contrast_fitter.print_single_contrast_metric(curr_i);
                }
                break;
            case '\'':
                if (key.shift) {
                    contrast_fitter.pupils[curr_i].params.radius *= 1.1;
                    contrast_fitter.pupils[curr_i].circle = contrast_fitter.circleFromParams(contrast_fitter.pupils[curr_i].params);
                    std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                    contrast_fitter.print_single_contrast_metric(curr_i);
                }
                break;

            case 'e':
                if (animate)
                    edge_fitter.refine_with_inliers(animate_func);
                else
                    edge_fitter.refine_with_inliers();
                recalc_ellipdist = true;
                display_est_pupil_ellipses_lm = true;

                break;

			case 'r':
                if (key.shift) {
                    if (!recording_writer.isOpened()) {
                        int fourcc = CV_FOURCC('M','J','P','G');
                        std::cout << "Start recording" << std::endl;
                        std::string out_file = (imagedir / "video.avi").string();
                        int done = recording_writer.open(
                            out_file,
                            fourcc,
                            25,
                            curr_disp.size());
                        if (!done) {
                            std::cerr << "Start failed!" << std::endl;
                        }
                    } else {
                        std::cout << "Stop recording" << std::endl;
                        recording_writer.release();
                    }
                } else {
                    ransac = !ransac;
                    if (ransac) {
                        std::cout << "RANSAC enabled" << std::endl;
                    } else {
                        std::cout << "RANSAC disabled" << std::endl;
                    }
                }
				break;
            case 'p':
                std::cout << curr_i << std::endl;
                std::cout << contrast_fitter.pupils[curr_i].params.theta << "," << contrast_fitter.pupils[curr_i].params.psi << "," <<  contrast_fitter.pupils[curr_i].params.radius << std::endl;
                contrast_fitter.print_single_contrast_metric(curr_i);
                if (curr_i < N) {
                    /*std::cout << EllipseGoodnessFunction<double>()(
                        contrast_fitter.eye,
                        contrast_fitter.pupils[curr_i].params.theta, contrast_fitter.pupils[curr_i].params.psi, contrast_fitter.pupils[curr_i].params.radius,
                        contrast_fitter.focal_length,
                        contrast_fitter.pupils[curr_i].observation.image) << std::endl;

                    std::cout << exp(-sq(contrast_fitter.pupils[curr_i].params.radius - 2.5)/sq(1.0)) << std::endl;

                    if (curr_i > 0) {
                        std::cout << angleDiffGoodness(
                            contrast_fitter.pupils[curr_i-1].params.theta, contrast_fitter.pupils[curr_i-1].params.psi,
                            contrast_fitter.pupils[curr_i].params.theta, contrast_fitter.pupils[curr_i].params.psi,
                            1) << std::endl;

                        std::cout << exp(-sq(contrast_fitter.pupils[curr_i].params.radius-contrast_fitter.pupils[curr_i-1].params.radius)/sq(1.0)) << std::endl;
                    }*/
                }
                break;

            default:
                std::cout << "Unknown key code: " << (int)key.code << " (ascii " << std::string(1, key.code) << ")" << std::endl;
                break;
			}
		}
	}

	//catch (std::exception& e) {
	//	std::cerr << e.what() << std::endl;
	//	while (cv::waitKey(10) == -1) {}
	//	return 1;
	//}

	return 0;
}

