1. Install Docker
2. Clone git - git clone https://github.com/zetsudan/optical-circuit-merg
3. go to directory - cd optical-circuit-merg
4. build container - docker build -t optical-circuit-merger:latest .
5. start container - docker run -d --name optical-circuit-merger   -p 8000:8000   --restart unless-stopped   optical-circuit-merger:latest

