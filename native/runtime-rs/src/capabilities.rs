#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Capability {
    FileRead,
    FileWrite,
    Network,
    Process,
    GitRead,
    GitWrite,
    ModelInfer,
    DebugEscalate,
    FfiCall,
}

impl Capability {
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "file.read" => Some(Self::FileRead),
            "file.write" => Some(Self::FileWrite),
            "network" => Some(Self::Network),
            "process" => Some(Self::Process),
            "git.read" => Some(Self::GitRead),
            "git.write" => Some(Self::GitWrite),
            "model.infer" => Some(Self::ModelInfer),
            "debug.escalate" => Some(Self::DebugEscalate),
            "ffi.call" => Some(Self::FfiCall),
            _ => None,
        }
    }
}
