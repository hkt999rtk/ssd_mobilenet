#include <iostream>
#include <filesystem>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
	if (argc < 4) {
		cout << "Usage: " << argv[0] << " [input image director] [output image directory] [output jason directory]" << endl;
		exit(1);
	}

	for (int i=1; i<4; i++) {
		if (!fs::is_directory(argv[i])) {
			cout << "Error: " << argv[i] << " is not a directory." << endl;
			exit(1);
		}
	}
	// travel input images
	const fs::path pathToShow{ argv[1]  };
	for (const auto& entry : fs::directory_iterator(pathToShow)) {
		const auto filenameStr = entry.path().filename().string();
		if (entry.is_regular_file()) {
			cout << "curl \"http://localhost:8120/detect?input=" << get_current_dir_name() << "/" << argv[1]
				<< "/" << filenameStr << "&output=" << get_current_dir_name() << "/" << argv[2] << "/detect-" << filenameStr
				<< "&x_left=0.1&x_right=0.08" << "\"" << endl;
		}
	}
}
