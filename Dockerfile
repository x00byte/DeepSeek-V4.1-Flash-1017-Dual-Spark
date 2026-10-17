FROM nvcr.io/nvidia/pytorch:26.07-py3
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1
RUN test -f /usr/include/infiniband/verbs.h || (apt-get update && apt-get install -y --no-install-recommends libibverbs-dev ibverbs-providers && rm -rf /var/lib/apt/lists/*)
RUN pip install --no-cache-dir --no-deps xgrammar==0.2.8 && pip install --no-cache-dir transformers==5.18.0
COPY build/tensorfold /opt/tensorfold
RUN pip install --no-deps --no-build-isolation /opt/tensorfold && python -c "import torch,triton,tensorfold.families.deepseek_v41; assert triton.__version__ == '3.7.1'; assert 'nv26.07' in torch.__version__; print(torch.__version__)"
ENTRYPOINT ["python", "-m", "tensorfold"]
