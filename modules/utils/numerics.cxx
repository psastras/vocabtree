#include "numerics.hpp"

namespace numerics {

	std::vector< std::pair<uint32_t, float> > sparsify(const cv::Mat &dense) {
		std::vector< std::pair<uint32_t, float> > sparse;
		for(int i=0; i<dense.size().area(); i++) {
			if( dense.at<float>(i) != 0.f) sparse.push_back( std::pair<uint32_t, float>(i, dense.at<float>(i)) );
		}
		return sparse;
	}

	float cos_sim(const std::vector<std::pair<uint32_t, float> > &weights0, 
		const std::vector<std::pair<uint32_t, float> > &weights1,
		const std::vector<float> &idfw) {
		float ab = 0.f, a2 = 0.f, b2 = 0.f;
		for(size_t i=0, j=0; i < weights0.size() || j < weights1.size();) {
			if(i < weights0.size() && j < weights1.size()) {
		    	if (weights0[i].first == weights1[j].first) {
		    		float a = weights0[i].second * idfw[weights0[i].first];
		    		float b = weights1[j].second * idfw[weights1[j].first];
		    		ab += a*b;
					a2 += a*a;
					b2 += b*b;
					i++, j++;
		    	} else if (weights0[i].first < weights1[j].first) {
		    		float a = weights0[i].second * idfw[weights0[i].first];
		    		a2 += a*a;
		    		i++;
		    	} else {
		    		float b = weights1[j].second * idfw[weights1[j].first];
		    		b2 += b*b;
		    		j++;
		    	}
	    	} else if(i < weights0.size()) {
	    		float a = weights0[i].second * idfw[weights0[i].first];
		    	a2 += a*a;
		    	i++;
	    	} else {
	    		float b = weights1[j].second * idfw[weights1[j].first];
		    	b2 += b*b;
		    	j++;
	    	}
		}
		return ab / (sqrtf(a2)*sqrtf(b2));
	}
}