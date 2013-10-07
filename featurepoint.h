
struct FeaturePoint{
	int x, y;
	float *feature_vector;
	int scale_level;
	FeaturePoint(int feature_size, int px, int py, int level){
		feature_vector = new float[feature_size];
		x = px; y = py;
		scale_level = level;
	};
}typedef FeaturePoint;