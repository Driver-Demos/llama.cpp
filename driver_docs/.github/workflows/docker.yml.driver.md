# Purpose
This file is a GitHub Actions workflow configuration file, written in YAML, designed to automate the process of building and publishing Docker images to Docker Hub. It provides a specific and narrow functionality focused on continuous integration and deployment (CI/CD) for Docker images. The workflow is triggered manually, on a schedule, or by specific events, and it includes steps for setting up the environment, logging into Docker Hub, and building and pushing different variants of Docker images (full, light, and server) for various platforms. The file is crucial to the codebase as it ensures that Docker images are consistently built and deployed, facilitating the development and deployment process by automating repetitive tasks and maintaining a reliable build pipeline.
# Content Summary
This GitHub Actions workflow is designed to automate the process of building and publishing Docker images to Docker Hub. The workflow is named "Publish Docker image" and can be triggered manually or scheduled to run daily at 4:12 AM UTC. The workflow uses third-party actions, which are not certified by GitHub, and developers are advised to pin these actions to a specific commit SHA to ensure consistency.

The workflow is configured to run on an Ubuntu 22.04 environment and utilizes a concurrency group to manage simultaneous runs, canceling any in-progress jobs if a new one is triggered. It grants write permissions to the GitHub packages, which is necessary for pushing Docker images.

The workflow defines a job named `push_to_registry`, which is responsible for building and pushing Docker images. It uses a matrix strategy to handle multiple configurations, each specifying a Dockerfile and target platforms. The configurations include different tags such as "cpu", "cuda", "musa", "intel", and "vulkan", each with specific build options. Notably, the "arm64" platform is excluded from some configurations due to known issues, and the "rocm" configuration is disabled due to a compiler error.

The job consists of several steps:

1. **Check out the repository**: Uses the `actions/checkout` action to fetch the repository's full history, which is necessary for determining the build number.

2. **Set up QEMU and Docker Buildx**: These steps prepare the environment for building multi-platform Docker images.

3. **Log in to Docker Hub**: Authenticates with Docker Hub using the GitHub repository owner's credentials and a GitHub token.

4. **Determine tag name**: A script calculates the Docker image tags based on the branch name, build number, and commit hash. It outputs tags for full, light, and server versions of the images.

5. **Free Disk Space**: Optionally frees up disk space on the runner if required by the configuration.

6. **Build and push Docker images**: Three separate steps build and push the full, light, and server versions of the Docker images. Each step uses the `docker/build-push-action` to build the image for the specified platforms and push it to Docker Hub. The workflow leverages GitHub's experimental cache to optimize the build process.

This workflow is a comprehensive solution for automating Docker image management, ensuring that images are consistently built and available on Docker Hub with appropriate versioning and tagging.
